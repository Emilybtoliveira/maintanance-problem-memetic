# Maintenance Planning Problem

Este projeto implementa um Algoritmo Memético para o Problema de Planejamento de Manutenções em Redes de Transmissão de Energia Elétrica.  
O ambiente é gerenciado com **Poetry** para controle de dependências e versões.

**Autores:** [Emily Oliveira](mailto:e291111@dac.unicamp.br)  e [Pedro Henrique](mailto:p295747@dac.unicamp.br)  

## Requisitos

Antes de começar, certifique-se de ter instalado:

- **Python 3.12** ou superior  
- **[Poetry](https://python-poetry.org/docs/#installation)**  
- **[Gurobi](https://www.gurobi.com/downloads/gurobi-software/)** com licença válida

---

## Instalação do ambiente

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/<seu-usuario>/<seu-repositorio>.git
cd maintanance-problem-memetic
poetry install
```

> O Poetry criará automaticamente um ambiente virtual e instalará todas as dependências definidas no `pyproject.toml`.

---

##  Executando o projeto

Ative o ambiente virtual do Poetry:

```bash
poetry shell
```

Em seguida, execute o script principal (por exemplo):

```bash
python python src/main.py
```


Se preferir rodar um comando sem ativar o shell:

```bash
poetry run python src/main.py
```

---

## Estrutura do projeto

```
maintanance-problem-memetic/
├── input                  # Arquivos de instâncias e de parâmetros
├── src/                  
│   ├── datamodels/        # Estruturas de dados
│   ├── solvers/           # Implementações do exato (Gurob) e AM
│   └── main.py    
└── utils/                 # Funções auxiliares
├── README.md              
└── pyproject.toml         
```

---

## Dependências principais

| Biblioteca  | Versão | Descrição |
|--------------|---------|------------|
| `gurobipy`   | ^12.0.3 | API Python para o solver Gurobi |
| `pydantic`   | ^2.12.3 | Validação e modelagem de dados |
| `numpy`      | ^2.3.4  | Cálculos numéricos e manipulação de vetores/matrizes |


