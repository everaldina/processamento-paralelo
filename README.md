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

- Nesta implementacao, ssim_kernel e lancado com um bloco por corte de B (blockIdx.x = indice do corte) - a ideia que voce sugeriu. Dentro de cada bloco, as threads dividem entre si os pixels validos daquele corte, cada uma calculando a janela win_size x win_size completa em volta do seu pixel - a mesma formula usada em metrics::calculate_ssim, so que em paralelo.
- Ao final, cada bloco faz uma reducao em memoria compartilhada (soma em arvore) para obter o SSIM medio daquele corte. Isso substitui o laco sequencial da CPU por paralelismo real dentro de um unico corte.
- Como cada bloco cuida de um corte inteiro de forma independente e existem centenas de cortes por volume, a GPU processa todos os cortes de B simultaneamente em uma unica chamada de kernel, em vez de um de cada vez como na CPU/OMP/MPI.
- Depois do kernel, o vetor com o SSIM de cada corte (poucos KB) e copiado de volta para a CPU, e um laco simples de host escolhe o maior valor (arg-max). E o equivalente ao MPI_MAXLOC da versao MPI, so que pequeno o suficiente para nao valer a pena fazer na GPU.

## Preparando o ambiente (antes de compilar)

Esta secao e para descobrir o que voce ja tem instalado e o que falta, com comandos pra rodar num terminal (funcionam tanto no Windows quanto no Linux/WSL, salvo onde indicado).

### Passo 1: conferir se a placa e o driver NVIDIA estao ok

Rode num terminal (cmd/PowerShell no Windows, bash no Linux ou dentro do WSL):

```bash
    nvidia-smi
```

Se o comando nao existir ("comando nao encontrado" / "not recognized"), o driver da NVIDIA nao esta instalado (ou nao esta no PATH) - ver "onde baixar" abaixo.

Se existir, a saida mostra uma tabela com o nome da GPU, a versao do driver (linha "Driver Version") e, no canto superior direito, "CUDA Version: X.Y" - esse numero e a versao MAXIMA do CUDA Toolkit que esse driver suporta rodar (nao e a versao do toolkit instalado, e so o teto permitido pelo driver). Se voce quiser instalar um CUDA Toolkit mais novo que esse numero, precisa atualizar o driver primeiro.

Onde baixar/atualizar o driver:
- Windows: https://www.nvidia.com/Download/index.aspx (escolha GeForce > <gpu name> > Windows) ou pelo app GeForce Experience, se ja tiver instalado.
- Linux (Ubuntu/Debian): `sudo ubuntu-drivers autoinstall` (detecta e instala o driver recomendado automaticamente) ou `sudo apt install nvidia-driver-<numero-da-versao>` para uma versao especifica.
- WSL2: NAO instale um driver Linux dentro do WSL - ele usa o driver que ja esta instalado no Windows (ver secao "Opcao C: WSL2" mais abaixo).

### Passo 2: conferir se voce ja tem o CUDA Toolkit (o nvcc)

```bash
    nvcc --version
```
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

    ```bash
    cl
    ```


   Se aparecer algo como "Microsoft (R) C/C++ Optimizing Compiler Version 19.3x.xxxxx for x64", esta tudo certo, pode pular pro passo A.3.

2. Alternativa: procure no menu Iniciar por "Visual Studio Installer". Se o app existir, abra-o - ele lista as instalacoes do Visual Studio/Build Tools que voce ja tem. Clique em "Modificar" em alguma delas e confira se o workload "Desenvolvimento para desktop com C++" esta marcado.

3. Alternativa por linha de comando, usando o `vswhere.exe` que a Microsoft instala automaticamente junto com qualquer Visual Studio/Build Tools (sempre no mesmo lugar, independente da versao):

    "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath

   Se esse comando imprimir um caminho (ex.: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools` ou `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools` - o nome da pasta varia por versao, as vezes e um ano, as vezes so um numero), voce ja tem o componente de C++ instalado. Se nao imprimir nada (linha vazia), falta instalar - ou o `vswhere.exe` nem existe nesse caminho, o que significa que nenhum Visual Studio/Build Tools foi instalado ainda.


Se esse atalho nao aparecer na busca do menu Iniciar (acontece as vezes com instalacoes so de Build Tools), rode o `vcvars64.bat` manualmente num `cmd.exe` comum (nao PowerShell - o `.bat` nao "gruda" as variaveis de ambiente na sessao do PowerShell), usando o caminho que apareceu no `vswhere.exe` do item 3: "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

Depois de rodar isso, teste `cl` de novo NA MESMA janela - so ai que ele fica disponivel.

Se nenhuma das checagens acima encontrar nada, instale:

1. Baixe em https://visualstudio.microsoft.com/downloads/ - role a pagina ate a secao "Tools for Visual Studio" e baixe "Build Tools for Visual Studio" (nao precisa da IDE completa, so as ferramentas de linha de comando - e gratuito, nao precisa de licenca paga pra esse uso).
2. Rode o instalador baixado. Na tela de selecao de workloads, marque "Desenvolvimento para desktop com C++" e clique em Instalar (pode demorar alguns minutos, baixa uns 2-4 GB).
3. Depois de terminar, vai aparecer "x64 Native Tools Command Prompt for VS 20xx" no menu Iniciar - abra e confira com `cl` (item 1 da lista de checagem acima).

#### A.3: conferir se ja tem o zlib compilado para MSVC (via vcpkg)

Isso e diferente do zlib que o MSYS2 ja tem instalado (usado no build normal do projeto, com `-lz`) - aquele e formato MinGW e o linker do MSVC nao consegue usar. Aqui a pergunta e se voce ja tem uma pasta do vcpkg com o zlib instalado dentro dela.

1. Veja se ja tem o vcpkg em algum lugar do seu computador. Se voce lembra de ja ter clonado antes, va direto pra pasta e rode:

    ```
    .\vcpkg\vcpkg list
    ```

   Se aparecer uma linha comecando com `zlib:x64-windows`, voce ja tem o zlib pronto pra MSVC e pode pular pro passo A.4 (so anote o caminho completo dessa pasta vcpkg, vai precisar dele no comando de compilacao).

   Se nao lembrar onde/se instalou, pode procurar pelo executavel com (pode demorar, procura o disco inteiro):
```
    where /r C:\ vcpkg.exe
```

2. Se nao tiver o vcpkg ainda, instale do zero. Rode:
```bash
    git clone https://github.com/microsoft/vcpkg
    .\vcpkg\bootstrap-vcpkg.bat
    .\vcpkg\vcpkg install zlib:x64-windows
```
   O ultimo comando demora alguns minutos (baixa e compila o zlib do zero). No final, confira que os tres arquivos abaixo existem (troque o caminho se voce clonou em outro lugar):

```bash
    dir vcpkg\installed\x64-windows\include\zlib.h
    dir vcpkg\installed\x64-windows\lib\zlib.lib
    dir vcpkg\installed\x64-windows\bin\zlib1.dll
```
   Se os tres aparecerem (sem erro de "Arquivo nao encontrado"), esta pronto para compilar.

   Problema comum: se so o `zlib.h` aparecer, mas `zlib.lib` e `zlib1.dll` derem "nao existe", a instalacao ficou pela metade (o vcpkg copiou o cabecalho mas nao terminou de compilar a biblioteca - acontece se o build for interrompido ou falhar no meio). Como o vcpkg pode ja ter marcado o pacote como "instalado" internamente, rodar `vcpkg install` de novo sozinho as vezes nao resolve; remova e reinstale para forcar do zero:

```
    .\vcpkg\vcpkg remove zlib:x64-windows
    .\vcpkg\vcpkg install zlib:x64-windows
```

   Deixe rodar ate o fim (alguns minutos) prestando atencao em qualquer erro no final da saida - se falhar de novo, geralmente aponta um arquivo de log dentro de `vcpkg\buildtrees\zlib\` com mais detalhes. Depois, confira os tres arquivos de novo.

#### A.4: compilar

1. Abra o "x64 Native Tools Command Prompt for VS" (o mesmo terminal do passo A.2) - precisa ser ESSE terminal especifico, nao um cmd/PowerShell comum, senao o `nvcc` nao vai achar o `cl.exe`.
2. Va ate a pasta do repositorio (troque pelo caminho real):

```
    cd /d C:\caminho\para\processamento-paralelo
```
3. Rode o comando de compilacao (troque CAMINHO_VCPKG pela pasta onde esta o vcpkg, por exemplo `C:\vcpkg` se voce clonou direto na raiz do C:):

    ```
    nvcc src\parallel_cuda.cu src\image_reader\mhd_reader.cpp -I./src -ICAMINHO_VCPKG\installed\x64-windows\include -LCAMINHO_VCPKG\installed\x64-windows\lib -lzlib -O3 -arch=sm_61 -o parallel_cuda.exe
    ```

   - Sao DOIS arquivos-fonte (parallel_cuda.cu e mhd_reader.cpp) na mesma chamada do nvcc - ver secao "Por que usa image_reader/" acima.
   - `-I./src` deixa os `#include "image_reader/mhd_reader.hpp"` e `#include "metrics/ssim_kernel.cuh"` (dentro de parallel_cuda.cu) serem encontrados.
   - `-arch=sm_61` e a compute capability da GTX 1060 (Pascal). Em outra GPU, troque pelo valor correspondente (ex.: sm_75 para Turing/RTX 20xx, sm_86 para Ampere/RTX 30xx).
   - Se o comando terminar sem mensagem de erro e criar o arquivo `parallel_cuda.exe` na pasta, funcionou. Confira com `dir parallel_cuda.exe`.
4. Copie `zlib1.dll` (de `CAMINHO_VCPKG\installed\x64-windows\bin\`) para a mesma pasta do `parallel_cuda.exe` (ou adicione essa pasta ao PATH) - o Windows precisa achar essa DLL em tempo de execucao, senao o programa da erro tipo "zlib1.dll not found" ao abrir:
```bash
    copy CAMINHO_VCPKG\installed\x64-windows\bin\zlib1.dll .
```
5. Se o nvcc reclamar que a versao do MSVC nao e oficialmente suportada, adicione `-allow-unsupported-compiler` ao comando (comum quando o Visual Studio instalado e mais novo que o CUDA Toolkit).

### Opcao B: Linux nativo (Ubuntu/Debian como exemplo)

Use esta opcao se estiver rodando direto num Linux com GPU NVIDIA (nao dentro do WSL - para isso, ver Opcao C).

1. Confira/instale o driver (Passo 1 acima).
2. Instale o CUDA Toolkit pelo repositorio oficial da NVIDIA (o exemplo abaixo e para Ubuntu 22.04 - troque a URL pela correspondente a sua distro/versao, disponivel na pagina de download do Passo 3):

```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install cuda-toolkit
```

3. Garanta que o nvcc esta no PATH (o instalador costuma pedir pra voce adicionar isso ao seu `~/.bashrc`; se `nvcc --version` nao funcionar depois de instalar, rode):

```bash
    export PATH=/usr/local/cuda/bin:$PATH
```

4. Instale as ferramentas de build e o zlib (aqui SIM da pra usar o zlib "normal" do sistema, porque no Linux o nvcc usa o g++ do proprio sistema como compilador hospedeiro - o problema de incompatibilidade so existe no Windows entre MinGW e MSVC):

```bash
    sudo apt install build-essential zlib1g-dev
```

5. Compile (na raiz do repositorio):
```
    nvcc src/parallel_cuda.cu src/image_reader/mhd_reader.cpp -I./src -O3 -arch=sm_61 -o parallel_cuda -lz
```

6. Rode: `./parallel_cuda -k 7 -t 256`

### Opcao C: WSL2 (Linux dentro do Windows) - evita precisar do Visual Studio

O WSL2 roda um Linux de verdade dentro do Windows, com acesso a GPU via um mecanismo da propria NVIDIA (nao precisa instalar nenhum driver Linux dentro dele - ele usa o driver do Windows que voce ja conferiu no Passo 1). Como o nvcc dentro do WSL usa g++ (o compilador hospedeiro oficialmente suportado no Linux), essa opcao evita toda a complicacao de MSVC/vcpkg da Opcao A.

1. Se ainda nao tem o WSL instalado, abra o PowerShell como administrador e rode:
```bash
    wsl --install -d Ubuntu
```
   Isso instala o WSL2 (se necessario) e uma distro Ubuntu. Reinicie o computador se for pedido. Se ja tiver alguma distro instalada, confira se ela e WSL2 (nao WSL1) com `wsl --list --verbose` - a coluna VERSION precisa mostrar 2.

2. Abra o terminal do Ubuntu (procure "Ubuntu" no menu Iniciar) e confira se a GPU aparece:
```bash
    nvidia-smi
```
   Se a GPU, o repasse da GPU do Windows pro WSL esta funcionando. Driver de video do Windows razoavelmente atual (praticamente qualquer um dos ultimos anos) ja inclui esse suporte - nao precisa fazer nada a mais alem de ja ter o driver do Passo 1 instalado no Windows.

3. Instale o CUDA Toolkit dentro do Ubuntu, usando o repositorio especifico "wsl-ubuntu" da NVIDIA (esse repositorio instala so o toolkit, sem tentar instalar um driver Linux por cima, que quebraria o repasse de GPU):

```bash
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install cuda-toolkit
```

4. Instale as ferramentas de build:
```bash
    sudo apt install build-essential zlib1g-dev git
```

5. Compile:
```bash
    nvcc src/parallel_cuda.cu src/image_reader/mhd_reader.cpp -I./src -O3 -arch=sm_61 -o parallel_cuda -lz
```

6. Atencao com os caminhos das imagens: `base_path_train`/`base_path_test`, dentro de `main()` em parallel_cuda.cu, estao escritos como `D:/workspace_data/tcc/orcascore/Challenge_data/...` (formato Windows). Dentro do WSL, o disco D: do Windows fica acessivel em `/mnt/d/`, entao pra rodar de dentro do WSL apontando pros dados que estao no Windows, troque esses dois caminhos por `/mnt/d/workspace_data/tcc/orcascore/Challenge_data/Training_set/Images` e `/mnt/d/workspace_data/tcc/orcascore/Challenge_data/Test_set/Images` antes de compilar (recompile depois de editar). Se copiar os dados pra dentro do filesystem do WSL (ex.: `~/dados/...`), a leitura tambem costuma ser mais rapida que via `/mnt/d/`.

7. Rode: `./parallel_cuda -k 7 -t 256`

## Como executar

No Windows (Opcao A), o binario gerado se chama `parallel_cuda.exe`; no Linux/WSL (Opcoes B e C), sem extensao, executado como `./parallel_cuda`. Os exemplos abaixo usam a forma Windows, so trocar pelo equivalente Linux/WSL.

parallel_cuda.exe [-k kernel_size] [-t threads_per_block] [-d device_id]

- -k / --kernel: tamanho da janela do SSIM, mesmo significado das outras versoes. Padrao: 7.
- -t / --threads: threads por bloco (precisa ser potencia de 2, ate 1024; se nao for, o programa arredonda para cima e avisa - ver secao "Como as threads, blocos e grids sao organizados"). Padrao: 256.
- -d / --device: indice da GPU a usar, se houver mais de uma. Padrao: 0.

Exemplo:

parallel_cuda.exe -k 7 -t 256

O log e salvo em output/cuda_t{threads_per_block}_k{kernel_size}.txt, no mesmo formato usado pelas outras versoes (ID da Imagem, Quantidade de cortes B avaliados, Tempo de Leitura Total (A+B), Tempo Medio Calculo SSIM (por slice), Tempo de Busca (SSIM loop), Tempo Total do Processo, Melhor Slice (Z), SSIM Maximo Encontrado), entao da para reaproveitar a mesma funcao parse_log ja usada nos notebooks existentes (por exemplo criando um notebooks/cuda_parallel.ipynb no mesmo estilo do omp_parallel.ipynb). Uma linha extra (Threads por Bloco) e gravada a mais - ela e ignorada pelo parse_log atual, mas da pra estender a funcao se quiser analisar o efeito do tamanho do bloco.

