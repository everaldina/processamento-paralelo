# Guia Rápido: Transição de C para C++

## 1. Estrutura de Pastas e Nomenclatura (Padrões C++)

Não existe um padrão oficial cravado em pedra no C++ (como existe no Go ou Rust), mas a comunidade adotou convenções muito fortes ao longo do tempo.

### Casing (Padrões de Nomeação)
* **Pastas/Diretórios:** Normalmente usa-se `snake_case` (ex: `core_engine`) ou `kebab-case` (ex: `core-engine`). Tudo minúsculo.
* **Arquivos Fonte (`.cpp`) e Headers (`.hpp` / `.h`):** * *Opção A (Estilo STL/Boost):* `snake_case.cpp` e `snake_case.hpp` (Mais comum na comunidade open-source e no próprio comitê do C++).
    * *Opção B (Estilo OO Clássico):* `PascalCase.cpp` e `PascalCase.hpp` (Comum em projetos de jogos, Unreal Engine ou empresas mais corporativas, espelhando o nome da Classe).
* **Extensões:**
    * Headers C++: Use `.hpp` (ou `.hh`, `.hxx`). Evite `.h` para não confundir com headers de C (embora muita gente ainda use `.h`).
    * Sources C++: Use `.cpp` (ou `.cc`, `.cxx`).

### Estruturação Clássica de Módulos / Diretórios
A separação clássica de um projeto C++ divide a interface (headers públicos) da implementação (sources).

```text
meu_projeto/
├── CMakeLists.txt        # Arquivo de build (CMake é o padrão de mercado)
├── include/              # Headers PÚBLICOS do seu projeto (os .hpp)
│   └── meu_modulo/       # Pasta com o nome do módulo/projeto
│       └── util.hpp
├── src/                  # Código fonte (.cpp) e headers PRIVADOS
│   ├── main.cpp
│   └── util.cpp
├── tests/                # Código de testes unitários (ex: GTest ou Catch2)
│   └── test_util.cpp
├── docs/                 # Documentação (arquivos Doxygen, Markdown)
└── external/             # Bibliotecas de terceiros (Third-party)
```

---

## 2. Principais Diferenças de Sintaxe: C vs C++

Você já sabe a base de C (ponteiros, loops, structs). Aqui estão os "upgrades" do C++ que mudam a sintaxe do dia a dia:

### A. Namespaces (Espaços de Nome)
No C, para evitar conflito de nomes, a gente criava prefixos tipo `opengl_render_box()`. No C++, usamos `namespace`.

```cpp
// Definição
namespace fisica {
    const float gravidade = 9.8f;
    void calcularQueda() { /* ... */ }
}

// Uso (utilizando o operador de resolução de escopo '::')
int main() {
    fisica::calcularQueda();
    float g = fisica::gravidade;
    return 0;
}
```

### B. Input/Output (E/S)
Você ainda pode usar `<stdio.h>` (`printf`/`scanf`), mas o C++ introduz a biblioteca `<iostream>`, baseada em streams.

```cpp
#include <iostream>
#include <string>

int main() {
    std::string nome; // String dinâmica do C++! Esqueça o char nome[50];
    
    // cout = console out, cin = console in
    std::cout << "Digite seu nome: "; 
    std::cin >> nome;
    
    std::cout << "Olá, " << nome << "!\n";
    return 0;
}
```

### C. Referências (`&`) vs Ponteiros (`*`)
O C só passa argumentos por valor ou por ponteiro. O C++ introduziu as **referências**, que funcionam como "ponteiros disfarçados" que não podem ser nulos e têm uma sintaxe bem mais limpa.

```cpp
// C style (Ponteiros)
void incrementa_c(int *valor) {
    (*valor)++;
}

// C++ style (Referências)
void incrementa_cpp(int &valor) {
    valor++; // Modifica a variável original sem precisar desreferenciar com *
}

int main() {
    int x = 10;
    incrementa_cpp(x); // Passa x diretamente, não precisa do &x
}
```

### D. Alocação de Memória
Esqueça `malloc`, `calloc` e `free`. C++ usa `new` e `delete` (ou melhor ainda, *Smart Pointers* se estiver usando C++11 em diante).

```cpp
// Jeito C++ Raiz (gerenciamento manual)
int* array = new int[10];
delete[] array; 

// Jeito C++ Moderno (Smart Pointers - Limpeza automática, esqueça memory leaks!)
#include <memory>

std::unique_ptr<int> valor = std::make_unique<int>(42);
// Não precisa dar free/delete. Quando sair do escopo, a memória é liberada.
```

### E. Structs são Classes
No C++, uma `struct` é exatamente a mesma coisa que uma `class`, com uma única diferença: numa struct tudo é público por padrão; numa class, tudo é privado por padrão. E ambas podem ter métodos (funções) dentro delas!

```cpp
struct Ponto {
    int x, y; // Público por padrão
    
    // Método dentro da struct
    void mover(int dx, int dy) {
        x += dx;
        y += dy;
    }
};
```

---

## 3. Guia de Documentação (Padrão Doxygen)

O padrão absoluto da indústria para documentar código C++ é o **Doxygen**. Ele lê seus comentários formatados e gera páginas HTML/PDF bonitas. A documentação deve ficar preferencialmente nos arquivos de cabeçalho (`.hpp`), onde os usuários da sua biblioteca vão olhar.

### Padrão de Comentários
Usa-se o bloco `/** ... */` ou o estilo de barra tripla `///`.

### Tags Mais Comuns:
* `@brief`: Resumo do que a função/classe faz.
* `@param`: Descreve um parâmetro (use `[in]`, `[out]`, ou `[in,out]`).
* `@return`: Descreve o que é retornado.
* `@throws` / `@exception`: Documenta se a função pode jogar alguma exceção.
* `@note`: Alguma observação importante para o desenvolvedor.

### Exemplo Prático de Documentação

```cpp
#ifndef MATEMATICA_GEOMETRIA_HPP
#define MATEMATICA_GEOMETRIA_HPP

namespace matematica {

    /**
     * @brief Calcula a área de um círculo.
     * * Esta função utiliza a constante matemática PI para determinar
     * a área baseada no raio fornecido.
     * * @param[in] raio O raio do círculo. Deve ser um valor positivo.
     * @return A área calculada do círculo. Retorna -1.0 se o raio for negativo.
     * * @note Precisão flutuante pode variar dependendo do hardware.
     */
    double calcular_area_circulo(double raio);

    /**
     * @brief Classe que representa um Retângulo bidimensional.
     */
    class Retangulo {
    public:
        /**
         * @brief Construtor padrão do Retângulo.
         * @param[in] largura Largura inicial.
         * @param[in] altura Altura inicial.
         */
        Retangulo(float largura, float altura);

        /// @brief Obtém a área do retângulo.
        /// @return Retorna a multiplicação da largura pela altura.
        float obter_area() const;
        
    private:
        float largura_;
        float altura_;
    };

} // namespace matematica

#endif // MATEMATICA_GEOMETRIA_HPP
```

## 4. Classes e Orientação a Objetos

No C, a gente agrupa dados em `structs` e cria funções separadas que recebem ponteiros para essas structs. No C++, dados e funções (métodos) vivem juntos dentro de uma `class`.

### Anatomia de uma Classe
A principal diferença de uma classe para uma struct é o **encapsulamento** (`public`, `private`, `protected`).

```cpp
// arquivo: Jogador.hpp (Declaração da classe)
#include <string>

class Jogador {
private:
    // Dados privados: Ninguém de fora da classe consegue acessar ou alterar diretamente.
    std::string nome_;
    int vida_;

public:
    // Construtor: Chamado automaticamente quando o objeto é criado
    Jogador(std::string nome, int vida_inicial);

    // Destrutor: Chamado automaticamente quando o objeto é destruído (sai de escopo)
    ~Jogador();

    // Métodos públicos (Interface para interagir com os dados privados)
    void receber_dano(int quantidade);
    int obter_vida() const; // O 'const' garante que esse método não altera nenhum dado da classe
};
```

### Implementação dos Métodos (O operador `::`)
Normalmente, a gente declara a classe no `.hpp` e implementa as funções no `.cpp`, usando o operador de resolução de escopo `::` para dizer a qual classe aquela função pertence.

```cpp
// arquivo: Jogador.cpp (Implementação)
#include "Jogador.hpp"
#include <iostream>

// Implementação do Construtor
Jogador::Jogador(std::string nome, int vida_inicial) {
    nome_ = nome;
    vida_ = vida_inicial;
    std::cout << "Jogador " << nome_ << " criado!\n";
}

// Implementação do Destrutor
Jogador::~Jogador() {
    std::cout << "Jogador " << nome_ << " foi destruido.\n";
}

// Implementação dos Métodos
void Jogador::receber_dano(int quantidade) {
    vida_ -= quantidade;
    if (vida_ < 0) vida_ = 0;
}

int Jogador::obter_vida() const {
    return vida_;
}
```

---

## 5. Sobrecarga de Funções (Function Overloading)

No C, você não pode ter duas funções com o mesmo nome (teria que fazer `print_int`, `print_float`, etc). No C++, o compilador é inteligente o suficiente para saber qual função chamar com base nos **argumentos** que você passa.

```cpp
#include <iostream>

void imprimir(int x) {
    std::cout << "Inteiro: " << x << "\n";
}

void imprimir(double x) {
    std::cout << "Double: " << x << "\n";
}

void imprimir(std::string x) {
    std::cout << "String: " << x << "\n";
}

int main() {
    imprimir(42);       // Chama a versão do int
    imprimir(3.14);     // Chama a versão do double
    imprimir("Olá");    // Chama a versão da string
    return 0;
}
```

---

## 6. Strings de Verdade (`std::string`)

Mexer com texto em C (`char*`, `strcpy`, `strcat`, `strcmp`) costuma dar muita dor de cabeça com buffer overflow e memória. O C++ resolve isso com a classe `std::string`.

```cpp
#include <iostream>
#include <string>

int main() {
    std::string saudacao = "Olá";
    std::string nome = "Mundo";

    // Concatenação fácil com o operador +
    std::string frase = saudacao + ", " + nome + "!";
    
    // Comparação direta (esqueça o strcmp!)
    if (saudacao == "Olá") {
        std::cout << "A string é igual!\n";
    }

    // Tamanho da string (esqueça o strlen!)
    std::cout << "Tamanho: " << frase.length() << "\n";

    return 0;
}
```

---

## 7. Coleções Dinâmicas (A maravilha do `std::vector`)

Se no C você precisava de um array que mudasse de tamanho, a solução era usar `malloc`, `realloc` e gerenciar tudo na mão. No C++, nós usamos os *Containers* da STL (Standard Template Library), sendo o `std::vector` o mais famoso. 

Um `vector` é um array dinâmico que cresce sozinho.

```cpp
#include <iostream>
#include <vector>

int main() {
    // Cria um vetor dinâmico de inteiros (vazio inicialmente)
    std::vector<int> numeros;

    // Adiciona elementos no final (ele aloca memória automaticamente)
    numeros.push_back(10);
    numeros.push_back(20);
    numeros.push_back(30);

    // Acesso seguro (tamanho)
    std::cout << "Tamanho do vetor: " << numeros.size() << "\n";

    // Acesso aos elementos (igual array de C)
    std::cout << "Primeiro elemento: " << numeros[0] << "\n";

    // O vetor limpa sua própria memória quando a função acaba!
    return 0;
}
```

---

## 8. Facilidades Modernas: `auto` e `Range-based for`

O C++ moderno (a partir do C++11) trouxe sintaxes muito mais limpas para o dia a dia.

### A palavra-chave `auto`
Se o compilador já sabe o tipo da variável pelo lado direito da igualdade, você não precisa digitar.

```cpp
auto x = 10;           // Compilador sabe que é int
auto pi = 3.1415f;     // Compilador sabe que é float
auto nome = "Maria";   // Compilador sabe que é const char*
```

### O For Baseado em Intervalo (Range-based for loop)
Iterar por arrays e vetores ficou muito mais parecido com linguagens como Python ou C#.

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> valores = {1, 2, 3, 4, 5};

    // Estilo C / C++ antigo
    for (size_t i = 0; i < valores.size(); i++) {
        std::cout << valores[i] << " ";
    }
    
    std::cout << "\n";

    // Estilo C++ Moderno (Range-based for)
    // Lê-se: "Para cada 'v' dentro de 'valores'..."
    for (int v : valores) {
        std::cout << v << " ";
    }

    return 0;
}
```

## 9. Gerenciamento de Memória: Stack vs Heap no C++

No C, a regra é clara: variáveis locais vão para a **Stack** (Pilha), e tudo que usa `malloc`/`calloc` vai para a **Heap** (Monte). No C++, a regra fundamental é a mesma, mas as abstrações mudam um pouco a forma como interagimos com a Heap.

### A. O Operador `new` e `delete` (Acesso Direto à Heap)

Sim, usar `new` no C++ é o equivalente direto a usar `malloc` no C. A memória alocada vai diretamente para a **Heap**.

A grande diferença é que o `new` não apenas separa os bytes na memória, mas ele também **chama o construtor** do objeto (inicializa os dados). O `delete` faz o inverso: chama o destrutor do objeto antes de liberar a memória (o equivalente ao `free`).

```cpp
void exemplo_new_delete() {
    // Aloca 1 inteiro na HEAP. 
    // O ponteiro 'ptr' em si vive na STACK, mas ele aponta para a HEAP.
    int* ptr = new int(42); 

    // Se você esquecer de chamar o delete, ocorre um Memory Leak!
    delete ptr; 
}
```

### B. Como o `std::vector` usa a memória (Metade Stack, Metade Heap)

O `std::vector` é brilhante porque ele esconde a complexidade da Heap de você, aplicando um conceito chamado **RAII** (Resource Acquisition Is Initialization).

Quando você cria um vetor localmente, o objeto vetor em si vive na **Stack**, mas os dados reais que você coloca dentro dele são alocados dinamicamente na **Heap**.

```cpp
#include <vector>

void exemplo_vetor() {
    // O objeto 'numeros' (que contém apenas uns 3 ponteiros de controle internamente)
    // é alocado na STACK.
    std::vector<int> numeros;

    // Quando você faz push_back, o vetor usa o 'new' (por debaixo dos panos)
    // para alocar espaço para o "10" lá na HEAP.
    numeros.push_back(10);
    numeros.push_back(20);

    // Quando a função acaba, a variável 'numeros' sai do escopo da STACK.
    // O destrutor do std::vector é chamado automaticamente e ele mesmo 
    // faz o 'delete' dos dados que estavam na HEAP. Zero vazamentos!
}
```

#### Visualizando a memória do `std::vector`:

```text
       STACK (Pilha)                      HEAP (Monte)
  +----------------------+            +----------------------+
  | std::vector<int> v;  |            | Array Dinâmico       |
  | - ptr_inicio   ----------(aponta)-----> [ 10 ]           |
  | - ptr_fim            |            |     [ 20 ]           |
  | - ptr_capacidade     |            |     [    ]           |
  +----------------------+            +----------------------+
```

### C. Por que evitar o `new`/`delete` manual no C++ Moderno?

No C++ moderno (C++11 em diante), a recomendação oficial é: **nunca use `new` e `delete` crus no seu código do dia a dia**. 

Para arrays dinâmicos, use `std::vector` ou `std::string`. Para objetos únicos dinâmicos na Heap, use **Smart Pointers** (`std::unique_ptr` ou `std::shared_ptr`). Eles alocam na Heap, mas liberam a memória automaticamente quando saem de escopo (exatamente como o vetor faz), prevenindo os temidos *memory leaks* que assombram os programadores de C.

```cpp
#include <memory> // Necessário para Smart Pointers

class Inimigo { /* ... */ };

void exemplo_smart_pointer() {
    // Cria um objeto Inimigo na HEAP.
    // O ptr_inimigo vive na STACK e gerencia a memória da HEAP sozinho.
    std::unique_ptr<Inimigo> ptr_inimigo = std::make_unique<Inimigo>();

    // Não precisa de 'delete ptr_inimigo;'.
    // Ao fim da função, a memória na Heap é liberada automaticamente!
}
```