# processamento-paralelo
Repositotio para disciplina de processamente paralelo


# How to run
Script usado como task de build para o projeto no vscode:

```json
{
	"version": "2.0.0",
	"tasks": [
		{
			"label": "build-main",
			"type": "shell",
			"command": "C:\\msys64\\ucrt64\\bin\\g++.exe",
			"args": [
				"src/sequencial.cpp",
				"src/image_display/save_image.cpp",
				"src/image_reader/mhd_reader.cpp", "src/metrics/ssim.cpp",
				"-I./src",
				"-O3",
				"-IC:/msys64/ucrt64/include/opencv4",
                "-LC:/msys64/ucrt64/lib",
				"-o",
				"sequencial.exe", "-lz",
				"-lopencv_core",
                "-lopencv_imgcodecs"
			],
			"options": {
				"cwd": "${workspaceFolder}"
			},
			"group": {
				"kind": "build",
				"isDefault": true
			},
			"problemMatcher": "$gcc"
		}
	]
}
```

# Implementação CUDA (GPU)

Versão do mesmo algoritmo (busca do corte de B mais parecido com o corte central de A via SSIM de janela deslizante) executada na GPU com NVIDIA CUDA. Arquivo: src/parallel_cuda.cu.

## Visão geral / mapeamento do problema para a GPU

- Terminologia CUDA: um kernel e a funcao que roda na GPU (marcada __global__); ele e lancado como uma grid (grade), composta por varios blocks (blocos), e cada bloco contem varias threads. __host__ marca codigo que roda normalmente na CPU (e o padrao quando nenhum qualificador e usado).
- Nesta implementacao, ssim_kernel e lancado com um bloco por corte de B (blockIdx.x = indice do corte) - a ideia que voce sugeriu. Dentro de cada bloco, as threads dividem entre si os pixels validos daquele corte, cada uma calculando a janela win_size x win_size completa em volta do seu pixel - a mesma formula usada em metrics::calculate_ssim, so que em paralelo.
- Ao final, cada bloco faz uma reducao em memoria compartilhada (soma em arvore) para obter o SSIM medio daquele corte. Isso substitui o laco sequencial da CPU por paralelismo real dentro de um unico corte.
- Como cada bloco cuida de um corte inteiro de forma independente e existem centenas de cortes por volume, a GPU processa todos os cortes de B simultaneamente em uma unica chamada de kernel, em vez de um de cada vez como na CPU/OMP/MPI.
- Depois do kernel, o vetor com o SSIM de cada corte (poucos KB) e copiado de volta para a CPU, e um laco simples de host escolhe o maior valor (arg-max). E o equivalente ao MPI_MAXLOC da versao MPI, so que pequeno o suficiente para nao valer a pena fazer na GPU.

## Como as threads, blocos e grids sao organizados

Esta secao detalha exatamente como threadIdx/blockIdx/blockDim se combinam no ssim_kernel, com numeros concretos. A hierarquia do CUDA e: thread (unidade de execucao) dentro de um block/bloco (grupo de threads que compartilham memoria compartilhada e podem sincronizar entre si) dentro de uma grid/grade (o conjunto de blocos lancados por uma chamada de kernel).

### A grid: um bloco por corte

O kernel e lancado assim:

  ssim_kernel<<<cortes_B, threads_per_block, shared_bytes>>>(...)

O primeiro numero entre os <<< >>> e o tamanho da grid (quantos blocos), o segundo e o tamanho do bloco (quantas threads por bloco), o terceiro e quantos bytes de memoria compartilhada dinamica cada bloco recebe. Aqui a grid tem sempre exatamente cortes_B blocos - um bloco por fatia do volume B, identificado dentro do kernel por blockIdx.x (0, 1, 2, ..., cortes_B - 1). Isso nao e configuravel por opcao de linha de comando porque se adapta sozinho ao volume: se B tem 300 cortes, a grid tem 300 blocos; se tiver 550, a grid tem 550 blocos.

Exemplo: para as imagens 512x512xslices que voce esta usando, se um par especifico tiver B com 400 cortes, o lancamento vira ssim_kernel<<<400, threads_per_block, ...>>> - 400 blocos, cada um calculando o SSIM medio entre a fatia central de A e uma das 400 fatias de B, todos ao mesmo tempo (sujeitos a quantos blocos a GPU consegue rodar simultaneamente - ver "ocupacao" mais abaixo).

### O bloco: threads_per_block e o round_up_pow2

O numero de threads por bloco vem do parametro -t (threads_per_block). Ele precisa ser uma potencia de 2 (1, 2, 4, 8, ..., 1024) por causa do algoritmo de reducao em memoria compartilhada usado no fim do kernel (explicado adiante) - esse algoritmo funciona dividindo o bloco ao meio repetidamente (256 -> 128 -> 64 -> ... -> 1), e isso so cobre todos os elementos sem sobra quando o total e potencia de 2.

E ai que entra round_up_pow2: se voce passar um -t que nao e potencia de 2, o programa arredonda para cima automaticamente e avisa no console, em vez de travar ou dar resultado errado. O algoritmo e simples (dobra a partir de 1 ate alcancar ou passar o valor pedido):

  int round_up_pow2(int v) {
      if (v < 1) v = 1;
      int p = 1;
      while (p < v) p <<= 1;
      return p;
  }

Exemplos de arredondamento:

  -t 200  -> 256   (128 < 200 <= 256)
  -t 256  -> 256   (ja e potencia de 2, fica igual)
  -t 300  -> 512
  -t 1000 -> 1024
  -t 1500 -> 2048, mas o main() em seguida limita a 1024 (maximo de threads por bloco permitido pela GPU - ver abaixo), entao o valor final usado e 1024

O limite de 1024 threads por bloco nao e escolha do programa, e uma limitacao de hardware: toda GPU NVIDIA desde Fermi (bem antes da GTX 1060) permite no maximo 1024 threads por bloco. A GTX 1060 (Pascal, compute capability 6.1) tambem permite ate 1024.

### Dividindo os pixels de um corte entre as threads do bloco (grid-stride loop)

Uma imagem 512x512 tem muito mais pixels validos do que threads em um bloco (no maximo 1024), entao cada thread processa varios pixels, um de cada vez, pulando de blockDim.x em blockDim.x. E o trecho:

  for (long long idx = threadIdx.x; idx < totalPixels; idx += blockDim.x) { ... }

threadIdx.x e o indice da thread dentro do bloco (0 até threads_per_block - 1); blockDim.x e o tamanho do bloco (o mesmo valor de threads_per_block). Entao a thread 0 processa os pixels 0, blockDim.x, 2*blockDim.x, ...; a thread 1 processa 1, blockDim.x+1, 2*blockDim.x+1, ...; e assim por diante, ate cobrir todo totalPixels.

totalPixels e o numero de pixels onde a janela do SSIM cabe inteira: para uma imagem width x height e janela win_size, com offset = win_size / 2, sobra uma borda de "offset" pixels de cada lado sem SSIM calculado (igual na versao sequencial), entao:

  validW = width  - 2 * offset
  validH = height - 2 * offset
  totalPixels = validW * validH   (0 se width/height forem menores que a janela)

Exemplo concreto, imagem 512x512 (o tamanho que voce esta usando) com kernel/janela win_size = 7 (offset = 3):

  validW = 512 - 6 = 506
  validH = 512 - 6 = 506
  totalPixels = 506 * 506 = 256.036 pixels

Com threads_per_block = 256: 256.036 / 256 = 1000,14..., ou seja, 256.036 = 256*1000 + 36. Isso significa que 36 threads do bloco passam 1001 vezes pelo laco (processam 1001 pixels cada) e as outras 220 threads passam 1000 vezes (a ultima passada delas simplesmente nao acontece porque idx já >= totalPixels). Ou seja, o trabalho fica bem distribuido, quase igual entre as 256 threads.

Com threads_per_block = 1024: 256.036 / 1024 = 250,03..., isto e, 256.036 = 1024*250 + 36 -> 36 threads fazem 251 passadas e as outras 988 fazem 250. Cada thread individualmente faz menos trabalho (menos pixels), mas ha 4x mais threads rodando ao mesmo tempo (dentro do limite de paralelismo real da GPU) e cada uma usa mais memoria compartilhada para a reducao (ver abaixo).

Se o kernel/janela fosse maior, por exemplo win_size = 9 (offset = 4): validW = validH = 512 - 8 = 504, totalPixels = 504*504 = 254.016 - um pouco menos pixels validos (a borda descartada e maior), mas cada pixel exige mais trabalho dentro do laco de janela (9x9 = 81 leituras em vez de 7x7 = 49), entao o kernel fica mais lento por pixel mesmo com pouco menos pixels totais - o mesmo efeito que -k ja tinha nas versoes sequencial/OMP/MPI.

### A reducao em memoria compartilhada (por que precisa ser potencia de 2)

Depois do laco acima, cada thread tem um localSum (a soma do SSIM dos pixels que ela processou). Para virar um unico "SSIM medio do corte", essas threads_per_block somas parciais precisam ser somadas entre si dentro do bloco. Isso e feito em memoria compartilhada (sdata[], visivel só para as threads do mesmo bloco, muito mais rapida que a memoria global da GPU):

  sdata[threadIdx.x] = localSum;
  __syncthreads();
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
      __syncthreads();
  }
  // sdata[0] agora tem a soma de todas as threads do bloco

Cada thread guarda seu resultado em sdata[threadIdx.x]. Depois, a cada rodada, metade das threads ainda ativas soma o valor "da outra metade" no seu proprio - por isso precisa ser potencia de 2, para a metade sempre dar um numero inteiro sem sobrar elemento nenhum. Exemplo com um bloco pequeno de 8 threads (sdata = [a0, a1, a2, a3, a4, a5, a6, a7]):

  s=4: sdata[0]+=sdata[4], sdata[1]+=sdata[5], sdata[2]+=sdata[6], sdata[3]+=sdata[7]
       -> sdata = [a0+a4, a1+a5, a2+a6, a3+a7, a4, a5, a6, a7] (so os 4 primeiros importam agora)
  s=2: sdata[0]+=sdata[2], sdata[1]+=sdata[3]
       -> sdata[0] = a0+a4+a2+a6, sdata[1] = a1+a5+a3+a7
  s=1: sdata[0]+=sdata[1]
       -> sdata[0] = a0+a1+a2+a3+a4+a5+a6+a7 (soma de todo mundo)

Sao log2(blockDim.x) rodadas (3 rodadas para 8 threads, 8 rodadas para 256, 10 rodadas para 1024). __syncthreads() entre as rodadas garante que nenhuma thread leia um valor que outra ainda nao escreveu. No fim, a thread 0 divide sdata[0] pelo totalPixels e grava o resultado (SSIM medio do corte) em ssimPerSlice[blockIdx.x].

O terceiro parametro do lancamento do kernel (shared_bytes) reserva o espaco desse sdata[] - e sempre threads_per_block * sizeof(double):

  -t 256  -> shared_bytes = 256  * 8 bytes = 2.048 bytes (2 KB)
  -t 1024 -> shared_bytes = 1024 * 8 bytes = 8.192 bytes (8 KB)

A GTX 1060 (Pascal) permite ate 48 KB de memoria compartilhada por bloco, entao mesmo no maximo de 1024 threads sobra bastante margem.

### Ocupacao: quantos blocos rodam ao mesmo tempo de verdade

A GTX 1060 tem 10 SMs (Streaming Multiprocessors, as "unidades" que executam blocos). Se a grid tem, por exemplo, 400 blocos (400 cortes em B), a GPU nao roda os 400 ao mesmo tempo - ela roda quantos couberem por SM simultaneamente (limitado por registradores e memoria compartilhada usados por bloco) e vai processando o restante assim que um SM libera espaco. Isso e automatico (o "scheduler" da GPU cuida disso), voce nao precisa configurar nada a mais - so importa saber que aumentar threads_per_block aumenta o uso de memoria compartilhada por bloco, o que pode reduzir quantos blocos cabem ao mesmo tempo por SM; e essa e uma das razoes pela qual vale testar valores diferentes de -t (256, 512, 1024, etc.) em vez de sempre usar o maximo.

Resumindo os tres numeros que aparecem no lancamento do kernel para o caso de uma imagem 512x512 com win_size=7 e -t 256:

  ssim_kernel<<<cortes_B, 256, 2048>>>(...)
             |          |    |
             |          |    +-- 2048 bytes de memoria compartilhada por bloco (256 * sizeof(double))
             |          +------- 256 threads por bloco, cada uma processando ~1000 dos 256.036 pixels validos
             +------------------ um bloco por corte de B (ex.: 400 blocos se B tiver 400 cortes)

## Por que usa image_reader/ (e como isso afeta a compilacao)

As suas imagens sao .zraw comprimidas com zlib (CompressedData=True), entao o programa precisa descomprimir os dados antes de calcular o SSIM. Em vez de reimplementar isso, parallel_cuda.cu reaproveita o mesmo leitor das outras versoes: image_reader/mhd_reader.hpp + mhd_reader.cpp (o mesmo arquivo usado por sequencial.cpp, parallel_slices_static.cpp e parallel_mpi.cpp), que ja sabe ler .mhd/.raw e .zraw com zlib.

O detalhe importante e que, no Windows, o nvcc usa o MSVC (cl.exe) como compilador hospedeiro para codigo C++ "normal" (nao-GPU) - ele nao tem suporte oficial ao MinGW/g++ do MSYS2 usado no build dos outros .cpp deste projeto (a task do VS Code no topo deste README). Objetos gerados pelo MinGW e pelo MSVC nao sao compativeis em ABI (STL, name mangling, CRT diferentes), entao nao da para simplesmente linkar o .o que o g++ gera de mhd_reader.cpp com o parallel_cuda.cu compilado pelo nvcc/MSVC.

A solucao nao e reimplementar a leitura (como numa versao anterior deste README), e sim mandar o nvcc compilar mhd_reader.cpp ele mesmo, na MESMA chamada em que compila parallel_cuda.cu - o nvcc aceita varios arquivos-fonte de uma vez e repassa os .cpp "puros" para o cl.exe automaticamente, entao tudo sai compilado pelo mesmo compilador/ABI (ver comando completo em "Como compilar" abaixo). Isso mantem os arquivos organizados (nao precisa duplicar a leitura MHD dentro do .cu) as custas de precisar apontar o zlib para o cl.exe tambem (proxima secao).

O kernel (ssim_kernel) fica num arquivo separado, src/metrics/ssim_kernel.cuh, ao lado da versao CPU (metrics/ssim.hpp/ssim_impl.hpp), em vez de estar dentro de parallel_cuda.cu. Isso e diferente do caso do mhd_reader: por ser so um header (.cuh) incluido via #include, nao vira um arquivo-fonte novo para o nvcc compilar separadamente - ele entra "colado" em parallel_cuda.cu no pre-processamento, do mesmo jeito que image_reader/mhd_reader.hpp ja e incluido. Ou seja, nao precisa adicionar nada ao comando de compilacao por causa disso: o -I./src que ja existe (para achar image_reader/mhd_reader.hpp) resolve o #include "metrics/ssim_kernel.cuh" tambem. Um .cu separado para o kernel (em vez de um .cuh) exigiria compilacao relocavel de device code (-rdc=true) e um passo extra de linkagem, entao nao foi esse o caminho escolhido.

## Preparando o ambiente (antes de compilar)

Esta secao e para descobrir o que voce ja tem instalado e o que falta, com comandos pra rodar num terminal (funcionam tanto no Windows quanto no Linux/WSL, salvo onde indicado).

### Passo 1: conferir se a placa e o driver NVIDIA estao ok

Rode num terminal (cmd/PowerShell no Windows, bash no Linux ou dentro do WSL):

    nvidia-smi

Se o comando nao existir ("comando nao encontrado" / "not recognized"), o driver da NVIDIA nao esta instalado (ou nao esta no PATH) - ver "onde baixar" abaixo.

Se existir, a saida mostra uma tabela com o nome da GPU (deve aparecer "GeForce GTX 1060"), a versao do driver (linha "Driver Version") e, no canto superior direito, "CUDA Version: X.Y" - esse numero e a versao MAXIMA do CUDA Toolkit que esse driver suporta rodar (nao e a versao do toolkit instalado, e so o teto permitido pelo driver). Se voce quiser instalar um CUDA Toolkit mais novo que esse numero, precisa atualizar o driver primeiro.

Para ver so a versao do driver, sem o resto da tabela:

    nvidia-smi --query-gpu=driver_version --format=csv,noheader

Onde baixar/atualizar o driver:
- Windows: https://www.nvidia.com/Download/index.aspx (escolha GeForce > GTX 1060 > Windows) ou pelo app GeForce Experience, se ja tiver instalado.
- Linux (Ubuntu/Debian): `sudo ubuntu-drivers autoinstall` (detecta e instala o driver recomendado automaticamente) ou `sudo apt install nvidia-driver-<numero-da-versao>` para uma versao especifica.
- WSL2: NAO instale um driver Linux dentro do WSL - ele usa o driver que ja esta instalado no Windows (ver secao "Opcao C: WSL2" mais abaixo).

### Passo 2: conferir se voce ja tem o CUDA Toolkit (o nvcc)

    nvcc --version

Se funcionar, mostra a versao instalada (algo como "Cuda compilation tools, release 12.4"). Se der erro de comando nao encontrado, o CUDA Toolkit nao esta instalado, ou esta instalado mas nao esta no PATH do terminal que voce abriu.

Checagem extra de onde ele esta (util se `nvcc --version` falhar mas voce lembra de ter instalado):
- Windows (PowerShell ou cmd): `where nvcc`
- Linux/WSL: `which nvcc`

No Windows, e comum o instalador do CUDA Toolkit colocar o nvcc em `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin` - se `where nvcc` nao achar nada mas essa pasta existir, falta adicionar ela ao PATH (ou usar um terminal que ja tenha isso configurado, como o instalador costuma oferecer).

### Passo 3: qual versao do CUDA Toolkit baixar

Pagina oficial de download: https://developer.nvidia.com/cuda-downloads (escolha seu sistema operacional e siga as opcoes de instalador que aparecem na propria pagina).

Regra pratica pra escolher a versao:
- Pegue o numero que apareceu em "CUDA Version" no `nvidia-smi` (passo 1) - essa e a versao mais nova que seu driver atual suporta.
- Baixe uma versao do CUDA Toolkit igual ou mais antiga que esse numero (se seu driver aceita ate CUDA 12.6, por exemplo, tanto o 12.6 quanto o 12.0 funcionam).
- Se quiser a versao mais nova do toolkit e ela for mais nova que o que seu driver aceita, atualize o driver primeiro (passo 1).

Para a GTX 1060 (Pascal, compute capability 6.1) especificamente: **NAO baixe direto a versao mais recente sem checar antes**. A NVIDIA vai removendo do compilador o suporte a arquiteturas antigas com o tempo - por exemplo, o CUDA Toolkit 13.x removeu completamente o suporte a Pascal (a mais antiga que o `nvcc` do 13.x aceita e `compute_75`, Turing). Prefira uma versao 12.x (essas ainda compilam pra Pascal). Depois de instalar, confira com:

    nvcc --list-gpu-arch

Se `sm_61` ou `compute_61` aparecer na lista, essa versao serve. Se nao aparecer nada da familia 6.x, precisa de uma versao mais antiga - o arquivo com todas as versoes anteriores fica em https://developer.nvidia.com/cuda-toolkit-archive.

Da pra ter varias versoes do CUDA Toolkit instaladas ao mesmo tempo (cada uma na sua pasta, ex.: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\` e `...\v13.3\`) sem conflito. So tome cuidado que digitar `nvcc` sozinho no terminal chama a versao que estiver primeiro no PATH, que pode nao ser a que voce quer - se isso for um problema, chame pelo caminho completo do `nvcc.exe` da versao certa (ex.: `"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"`) em vez de so `nvcc` nos comandos de compilacao abaixo. Ao instalar uma versao adicional, use a opcao de instalacao "Custom" e desmarque o componente de driver de video, se voce ja tiver um driver mais novo instalado - so instale os componentes do proprio toolkit.

## Como compilar

Escolha UMA das tres opcoes abaixo. As tres geram o mesmo programa; a diferenca e o sistema operacional/toolchain usado. Se voce quer evitar instalar o Visual Studio, a Opcao C (WSL2) e provavelmente a mais simples, porque usa o mesmo tipo de comando (apt + g++) que voce ja deve estar acostumado em Linux, sem a complicacao de MSVC/vcpkg.

### Opcao A: Windows nativo (MSVC)

Use esta opcao se quiser compilar direto no Windows, sem WSL. Ela tem mais passos que as outras porque o nvcc no Windows depende de duas coisas que talvez voce ainda nao tenha: o compilador da Microsoft (cl.exe, que vem com o "Visual Studio Build Tools") e uma copia do zlib compilada especificamente pra esse compilador. Os passos A.2 e A.3 abaixo sao so pra CONFERIR se voce ja tem cada coisa antes de instalar de novo.

#### A.1: conferir o CUDA Toolkit

Ja coberto no Passo 2 da secao anterior (`nvcc --version`). Se ainda nao tem, instale antes de continuar (Passo 3).

#### A.2: conferir se ja tem o Visual Studio Build Tools (cl.exe)

Atencao com uma pegadinha comum aqui: o `cl.exe` NAO fica disponivel num terminal comum (cmd/PowerShell abertos do jeito normal), mesmo quando ja esta instalado - ele so aparece dentro de um terminal especial que o proprio Visual Studio configura. Ou seja, se voce abrir um PowerShell qualquer e rodar `where cl`, o mais provavel e dar "nao encontrado" mesmo que tudo esteja instalado - isso sozinho NAO significa que falta instalar nada.

Formas de conferir se voce ja tem o Build Tools instalado:

1. Procure no menu Iniciar por "x64 Native Tools". Se aparecer um atalho tipo "x64 Native Tools Command Prompt for VS 2022" (ou 2019), voce ja tem o Build Tools instalado. Abra esse atalho (ele e diferente de um cmd normal - ja vem com o `cl.exe` configurado no PATH) e rode dentro dele:

    ```
    > cl
    ```


   Se aparecer algo como "Microsoft (R) C/C++ Optimizing Compiler Version 19.3x.xxxxx for x64", esta tudo certo, pode pular pro passo A.3.

2. Alternativa: procure no menu Iniciar por "Visual Studio Installer". Se o app existir, abra-o - ele lista as instalacoes do Visual Studio/Build Tools que voce ja tem. Clique em "Modificar" em alguma delas e confira se o workload "Desenvolvimento para desktop com C++" esta marcado.

3. Alternativa por linha de comando, usando o `vswhere.exe` que a Microsoft instala automaticamente junto com qualquer Visual Studio/Build Tools (sempre no mesmo lugar, independente da versao):

    "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath

   Se esse comando imprimir um caminho (ex.: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` ou `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools` - o nome da pasta varia por versao, as vezes e um ano, as vezes so um numero), voce ja tem o componente de C++ instalado. Se nao imprimir nada (linha vazia), falta instalar - ou o `vswhere.exe` nem existe nesse caminho, o que significa que nenhum Visual Studio/Build Tools foi instalado ainda.

Cuidado com um erro comum: **navegar de proposito ate a pasta do BuildTools num cmd normal e digitar `cl` la dentro nao funciona**, mesmo se a pasta existir (o `cl.exe` fica varios subniveis mais fundo, tipo `...\BuildTools\VC\Tools\MSVC\<versao>\bin\Hostx64\x64\cl.exe`, e mesmo estando na pasta certa faltariam as variaveis de ambiente que o `cl.exe` precisa pra achar os headers/bibliotecas do Windows SDK). O jeito certo e sempre abrir o atalho "x64 Native Tools Command Prompt", que roda um script (`vcvars64.bat`) configurando tudo isso automaticamente antes de te devolver o prompt.

Se esse atalho nao aparecer na busca do menu Iniciar (acontece as vezes com instalacoes so de Build Tools), rode o `vcvars64.bat` manualmente num `cmd.exe` comum (nao PowerShell - o `.bat` nao "gruda" as variaveis de ambiente na sessao do PowerShell), usando o caminho que apareceu no `vswhere.exe` do item 3:

    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

Depois de rodar isso, teste `cl` de novo NA MESMA janela - so ai que ele fica disponivel.

Se nenhuma das checagens acima encontrar nada, instale:

1. Baixe em https://visualstudio.microsoft.com/downloads/ - role a pagina ate a secao "Tools for Visual Studio" e baixe "Build Tools for Visual Studio" (nao precisa da IDE completa, so as ferramentas de linha de comando - e gratuito, nao precisa de licenca paga pra esse uso).
2. Rode o instalador baixado. Na tela de selecao de workloads, marque "Desenvolvimento para desktop com C++" e clique em Instalar (pode demorar alguns minutos, baixa uns 2-4 GB).
3. Depois de terminar, vai aparecer "x64 Native Tools Command Prompt for VS 20xx" no menu Iniciar - abra e confira com `cl` (item 1 da lista de checagem acima).

#### A.3: conferir se ja tem o zlib compilado para MSVC (via vcpkg)

Isso e diferente do zlib que o MSYS2 ja tem instalado (usado no build normal do projeto, com `-lz`) - aquele e formato MinGW e o linker do MSVC nao consegue usar. Aqui a pergunta e se voce ja tem uma pasta do vcpkg com o zlib instalado dentro dela.

1. Veja se ja tem o vcpkg em algum lugar do seu computador. Se voce lembra de ja ter clonado antes, va direto pra pasta e rode:

    .\vcpkg\vcpkg list

   Se aparecer uma linha comecando com `zlib:x64-windows`, voce ja tem o zlib pronto pra MSVC e pode pular pro passo A.4 (so anote o caminho completo dessa pasta vcpkg, vai precisar dele no comando de compilacao).

   Se nao lembrar onde/se instalou, pode procurar pelo executavel com (pode demorar, procura o disco inteiro):

    where /r C:\ vcpkg.exe

2. Se nao tiver o vcpkg ainda, instale do zero. Primeiro confira se tem o `git` (a maioria de quem programa em Windows ja tem, mas confira):

    git --version

   Se nao tiver, baixe em https://git-scm.com/downloads e instale antes de continuar. Com o git disponivel, num terminal normal (nao precisa ser o "x64 Native Tools"), na pasta onde quiser deixar o vcpkg (ex.: `C:\`):

    git clone https://github.com/microsoft/vcpkg
    .\vcpkg\bootstrap-vcpkg.bat
    .\vcpkg\vcpkg install zlib:x64-windows

   O ultimo comando demora alguns minutos (baixa e compila o zlib do zero). No final, confira que os tres arquivos abaixo existem (troque o caminho se voce clonou em outro lugar):

    dir vcpkg\installed\x64-windows\include\zlib.h
    dir vcpkg\installed\x64-windows\lib\zlib.lib
    dir vcpkg\installed\x64-windows\bin\zlib1.dll

   Se os tres aparecerem (sem erro de "Arquivo nao encontrado"), esta pronto para compilar.

   Problema comum: se so o `zlib.h` aparecer, mas `zlib.lib` e `zlib1.dll` derem "nao existe", a instalacao ficou pela metade (o vcpkg copiou o cabecalho mas nao terminou de compilar a biblioteca - acontece se o build for interrompido ou falhar no meio). Como o vcpkg pode ja ter marcado o pacote como "instalado" internamente, rodar `vcpkg install` de novo sozinho as vezes nao resolve; remova e reinstale para forcar do zero:

    .\vcpkg\vcpkg remove zlib:x64-windows
    .\vcpkg\vcpkg install zlib:x64-windows

   Deixe rodar ate o fim (alguns minutos) prestando atencao em qualquer erro no final da saida - se falhar de novo, geralmente aponta um arquivo de log dentro de `vcpkg\buildtrees\zlib\` com mais detalhes. Depois, confira os tres arquivos de novo.

#### A.4: compilar

1. Abra o "x64 Native Tools Command Prompt for VS" (o mesmo terminal do passo A.2) - precisa ser ESSE terminal especifico, nao um cmd/PowerShell comum, senao o `nvcc` nao vai achar o `cl.exe`.
2. Va ate a pasta do repositorio (troque pelo caminho real):

    cd /d C:\caminho\para\processamento-paralelo

3. Rode o comando de compilacao (troque CAMINHO_VCPKG pela pasta onde esta o vcpkg, por exemplo `C:\vcpkg` se voce clonou direto na raiz do C:):

    ```
    nvcc src\parallel_cuda.cu src\image_reader\mhd_reader.cpp -I./src -ICAMINHO_VCPKG\installed\x64-windows\include -LCAMINHO_VCPKG\installed\x64-windows\lib -lzlib -O3 -arch=sm_61 -o parallel_cuda.exe
    ```

   - Sao DOIS arquivos-fonte (parallel_cuda.cu e mhd_reader.cpp) na mesma chamada do nvcc - ver secao "Por que usa image_reader/" acima.
   - `-I./src` deixa os `#include "image_reader/mhd_reader.hpp"` e `#include "metrics/ssim_kernel.cuh"` (dentro de parallel_cuda.cu) serem encontrados.
   - `-arch=sm_61` e a compute capability da GTX 1060 (Pascal). Em outra GPU, troque pelo valor correspondente (ex.: sm_75 para Turing/RTX 20xx, sm_86 para Ampere/RTX 30xx).
   - Se o comando terminar sem mensagem de erro e criar o arquivo `parallel_cuda.exe` na pasta, funcionou. Confira com `dir parallel_cuda.exe`.
4. Copie `zlib1.dll` (de `CAMINHO_VCPKG\installed\x64-windows\bin\`) para a mesma pasta do `parallel_cuda.exe` (ou adicione essa pasta ao PATH) - o Windows precisa achar essa DLL em tempo de execucao, senao o programa da erro tipo "zlib1.dll not found" ao abrir:

    copy CAMINHO_VCPKG\installed\x64-windows\bin\zlib1.dll .

5. Se o nvcc reclamar que a versao do MSVC nao e oficialmente suportada, adicione `-allow-unsupported-compiler` ao comando (comum quando o Visual Studio instalado e mais novo que o CUDA Toolkit).

### Opcao B: Linux nativo (Ubuntu/Debian como exemplo)

Use esta opcao se estiver rodando direto num Linux com GPU NVIDIA (nao dentro do WSL - para isso, ver Opcao C).

1. Confira/instale o driver (Passo 1 acima).
2. Instale o CUDA Toolkit pelo repositorio oficial da NVIDIA (o exemplo abaixo e para Ubuntu 22.04 - troque a URL pela correspondente a sua distro/versao, disponivel na pagina de download do Passo 3):

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install cuda-toolkit

3. Garanta que o nvcc esta no PATH (o instalador costuma pedir pra voce adicionar isso ao seu `~/.bashrc`; se `nvcc --version` nao funcionar depois de instalar, rode):

    export PATH=/usr/local/cuda/bin:$PATH

4. Instale as ferramentas de build e o zlib (aqui SIM da pra usar o zlib "normal" do sistema, porque no Linux o nvcc usa o g++ do proprio sistema como compilador hospedeiro - o problema de incompatibilidade so existe no Windows entre MinGW e MSVC):

    sudo apt install build-essential zlib1g-dev

5. Compile (na raiz do repositorio):

    nvcc src/parallel_cuda.cu src/image_reader/mhd_reader.cpp -I./src -O3 -arch=sm_61 -o parallel_cuda -lz

6. Rode:

    ./parallel_cuda -k 7 -t 256

### Opcao C: WSL2 (Linux dentro do Windows) - evita precisar do Visual Studio

O WSL2 roda um Linux de verdade dentro do Windows, com acesso a GPU via um mecanismo da propria NVIDIA (nao precisa instalar nenhum driver Linux dentro dele - ele usa o driver do Windows que voce ja conferiu no Passo 1). Como o nvcc dentro do WSL usa g++ (o compilador hospedeiro oficialmente suportado no Linux), essa opcao evita toda a complicacao de MSVC/vcpkg da Opcao A.

1. Se ainda nao tem o WSL instalado, abra o PowerShell como administrador e rode:

    wsl --install -d Ubuntu

   Isso instala o WSL2 (se necessario) e uma distro Ubuntu. Reinicie o computador se for pedido. Se ja tiver alguma distro instalada, confira se ela e WSL2 (nao WSL1) com `wsl --list --verbose` - a coluna VERSION precisa mostrar 2.

2. Abra o terminal do Ubuntu (procure "Ubuntu" no menu Iniciar) e confira se a GPU aparece:

    nvidia-smi

   Se a GTX 1060 aparecer normalmente (mesma saida do Passo 1), o repasse da GPU do Windows pro WSL esta funcionando. Driver de video do Windows razoavelmente atual (praticamente qualquer um dos ultimos anos) ja inclui esse suporte - nao precisa fazer nada a mais alem de ja ter o driver do Passo 1 instalado no Windows.

3. Instale o CUDA Toolkit dentro do Ubuntu, usando o repositorio especifico "wsl-ubuntu" da NVIDIA (esse repositorio instala so o toolkit, sem tentar instalar um driver Linux por cima, que quebraria o repasse de GPU):

    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install cuda-toolkit

4. Instale as ferramentas de build:

    sudo apt install build-essential zlib1g-dev git

5. Pegue o codigo dentro do WSL. O jeito mais rapido de I/O e clonar o repositorio direto no filesystem do Linux (ex.: `~/processamento-paralelo`), em vez de acessar os arquivos do Windows por `/mnt/c/...` (que e bem mais lento):

    git clone <url-do-seu-repositorio> ~/processamento-paralelo
    cd ~/processamento-paralelo

6. Compile:

    nvcc src/parallel_cuda.cu src/image_reader/mhd_reader.cpp -I./src -O3 -arch=sm_61 -o parallel_cuda -lz

7. Atencao com os caminhos das imagens: `base_path_train`/`base_path_test`, dentro de `main()` em parallel_cuda.cu, estao escritos como `D:/workspace_data/tcc/orcascore/Challenge_data/...` (formato Windows). Dentro do WSL, o disco D: do Windows fica acessivel em `/mnt/d/`, entao pra rodar de dentro do WSL apontando pros dados que estao no Windows, troque esses dois caminhos por `/mnt/d/workspace_data/tcc/orcascore/Challenge_data/Training_set/Images` e `/mnt/d/workspace_data/tcc/orcascore/Challenge_data/Test_set/Images` antes de compilar (recompile depois de editar). Se copiar os dados pra dentro do filesystem do WSL (ex.: `~/dados/...`), a leitura tambem costuma ser mais rapida que via `/mnt/d/`.

8. Rode:

    ./parallel_cuda -k 7 -t 256

## Como executar

No Windows (Opcao A), o binario gerado se chama `parallel_cuda.exe`; no Linux/WSL (Opcoes B e C), sem extensao, executado como `./parallel_cuda`. Os exemplos abaixo usam a forma Windows, so trocar pelo equivalente Linux/WSL.

parallel_cuda.exe [-k kernel_size] [-t threads_per_block] [-d device_id]

- -k / --kernel: tamanho da janela do SSIM, mesmo significado das outras versoes. Padrao: 7.
- -t / --threads: threads por bloco (precisa ser potencia de 2, ate 1024; se nao for, o programa arredonda para cima e avisa - ver secao "Como as threads, blocos e grids sao organizados"). Padrao: 256.
- -d / --device: indice da GPU a usar, se houver mais de uma. Padrao: 0.

Exemplo:

parallel_cuda.exe -k 7 -t 256

O log e salvo em output/cuda_t{threads_per_block}_k{kernel_size}.txt, no mesmo formato usado pelas outras versoes (ID da Imagem, Quantidade de cortes B avaliados, Tempo de Leitura Total (A+B), Tempo Medio Calculo SSIM (por slice), Tempo de Busca (SSIM loop), Tempo Total do Processo, Melhor Slice (Z), SSIM Maximo Encontrado), entao da para reaproveitar a mesma funcao parse_log ja usada nos notebooks existentes (por exemplo criando um notebooks/cuda_parallel.ipynb no mesmo estilo do omp_parallel.ipynb). Uma linha extra (Threads por Bloco) e gravada a mais - ela e ignorada pelo parse_log atual, mas da pra estender a funcao se quiser analisar o efeito do tamanho do bloco.

## Memoria da GPU (VRAM)

O programa consulta a memoria da GPU sozinho, em dois pontos, usando cudaMemGetInfo(&livre, &total) (a funcao da API CUDA que responde "quanto tem no total e quanto esta livre agora"):

- No inicio do main(), imprime no console quanto de VRAM esta livre/total antes de comecar a bateria de testes.
- Dentro de process_image_pair, antes de alocar o volume B na GPU, confere se cabe na memoria livre; se nao couber, imprime um aviso e pula esse par (em vez de deixar o cudaMalloc falhar de forma menos clara).

Se quiser conferir a VRAM disponivel de fora do programa, o jeito mais simples e abrir um terminal e rodar:

    nvidia-smi

que mostra (entre outras coisas) quanto de memoria a GPU tem no total e quanto esta em uso no momento.

Sobre suas imagens especificamente (512 x 512 x numero de cortes, 16 bits por voxel): cada corte ocupa 512*512*2 bytes = 512 KB, entao um volume B com, digamos, 600 cortes ocupa 600*512 KB = ~300 MB na GPU - bem abaixo dos 6 GB da GTX 1060. Mesmo um volume (hipotetico) de 4000 cortes ficaria perto de 2 GB, ainda folgado. Ou seja, para o tamanho de imagem que voce esta usando isso nao deve ser um problema na pratica; a checagem automatica esta ai mais por seguranca/clareza do erro do que porque se espera que ela dispare.

## O que da para variar nos testes

Assim como a versao OMP variou threads e tamanho do bloco de agendamento, e a MPI variou o numero de processos, na versao CUDA da pra variar:

- kernel/janela do SSIM (-k): o mesmo teste ja feito nas outras versoes - quanto maior a janela, mais trabalho por pixel (ver exemplo numerico na secao sobre grid-stride loop acima).
- threads por bloco (-t): e o parametro mais proximo do numero de threads/processos testado nas outras versoes. Valores tipicos para experimentar: 32, 64, 128, 256, 512, 1024. Ele muda quantas threads dividem o trabalho de um unico corte e a ocupacao da GPU (poucas threads por bloco deixam a GPU ociosa; threads demais nem sempre ajudam, porque o gargalo pode passar a ser acesso a memoria ou porque menos blocos cabem ao mesmo tempo por SM).
- O numero de blocos por chamada de kernel nao e configuravel: e sempre igual ao numero de cortes de B, porque a ideia e "um bloco calcula um corte". Isso ja se adapta automaticamente ao tamanho do volume de cada par de imagens, sem precisar de um parametro extra.
- K1, K2 (as constantes da formula do SSIM) e L (a faixa dinamica, 65535 para as imagens de 16 bits deste projeto) ficam fixas de proposito no codigo, nao sao expostas como parametro - nao fazem parte do que este trabalho se propoe a testar, entao nao ha motivo pra transformar em opcao de linha de comando.

Para gerar varios arquivos de log (um por combinacao), rode o executavel varias vezes trocando -t (e opcionalmente -k), do mesmo jeito que ja existem varios slices_static com threads e kernels diferentes.

## Notas sobre corretude / comparacao com as outras versoes

- As somas da janela deslizante sao somas de valores inteiros (short) representados em double; para os tamanhos de janela usados aqui (ate 9x9) esses valores nunca chegam perto do limite de precisao exata de um double (2 elevado a 53), entao essa parte da conta e identica bit a bit a CPU, independente da ordem da soma.
- Ja a soma final do SSIM de todos os pixels de um corte e feita em paralelo (reducao em arvore dentro do bloco), enquanto a CPU faz essa soma sequencialmente. Soma de ponto flutuante nao e associativa, entao o SSIM de um corte pode diferir da CPU la pela decima segunda ou decima terceira casa decimal. E uma caracteristica normal de reducoes paralelas em GPU, nao um bug.
- Essa diferenca e muito menor que a tolerancia (tolerancia_teste = 1e-3) ja usada nos notebooks (funcao teste_corretude) para comparar sequencial vs. OMP/MPI, entao a comparacao de corretude deve passar normalmente. O corte escolhido como melhor (Melhor Slice Z) deve ser sempre o mesmo, ja que a diferenca de SSIM e muitas ordens de grandeza menor que a distancia entre o melhor corte e os demais.
- O campo Tempo Medio Calculo SSIM (por slice) tem um significado um pouco diferente aqui: nas versoes sequencial/OMP/MPI ele e a soma dos tempos individuais de cada calculo de SSIM dividida pela quantidade de cortes (custo intrinseco de uma SSIM, independente do grau de paralelismo). Na versao CUDA todos os cortes sao calculados numa unica chamada de kernel, entao nao existe uma medicao isolada por corte - o valor reportado e o tempo de busca dividido pelos cortes avaliados, ou seja, um throughput medio, nao um custo intrinseco. Leve essa ressalva em conta ao comparar esse campo especifico entre CPU e GPU; ja o tempo de busca e o tempo total do processo continuam diretamente comparaveis entre todas as versoes.

## Limitacoes conhecidas

- Usa uma unica GPU por execucao (-d escolhe qual, mas nao ha suporte a multiplas GPUs simultaneas).
- O volume B inteiro e copiado de uma vez para a memoria da GPU (sem dividir em pedacos). Como visto na secao "Memoria da GPU" acima, isso e folgado para o tamanho de imagem usado neste projeto (512x512xslices); volumes muito maiores que a VRAM disponivel exigiriam dividir a copia/kernel em pedacos (nao implementado, e o programa agora avisa e pula o par em vez de travar se isso acontecer).
- O zlib usado para compilar precisa ser uma copia separada compativel com MSVC (via vcpkg, por exemplo) - nao da pra reaproveitar a instalacao do MSYS2 usada pelos outros builds deste projeto (ver "Requisitos").
