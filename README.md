# IC-Autoloss[C4AI]
Este repositório é dedicado à documentação do projeto de pesquisa entitulado "AutoLoss: Descobrindo Funções de Perda de Maneira Automática", orientado pelo Prof. Dr Artur Jordão do Departamento de Engenharia de Computação e Sistemas Digitais (PCS) da Poli-usp 

### Sobre o Projeto

Durante a pesquisa, desenvolvemos três algoritmos inspirados no trabalho da disciplina Aprendizado Profundo - Redes Neurais (PCS5022), por Gustavo Nascimento e Leandro Mugnain. O objetivo principal foi expandir e refinar a busca por funções de perda (loss functions) com maior potencial de desempenho em modelos de aprendizado profundo, com ênfase em forecasting.

### Geração de Funções de Perda

Criamos um algoritmo de geração pseudo-aleatória de funções de perda diferenciáveis, estruturadas como grafos computacionais. Em vez da programação genética tradicional (baseada em mutações e seleção por torneio), utilizamos uma abordagem que explora um combinações matemáticas baseadas em funções primitivas diferenciaveis.
Foram aplicados filtros de rejeição para garantir: Existência de ŷ e y nas expressões; Convergência inicial ao treinar um modelo-teste.

### Fusão de Modelos com Diferentes Funções de Perda

implementamos um algoritmo para combinar o conhecimento adquirido com diferentes funções de perda. A estratégia consiste em:
	1.	Treinar um modelo de referência (ex: com MSE).
	2.	Treinar versões alternativas com outras funções de perda.
	3.	Calcular os deltas de pesos (Δ = w_novo - w_base) de cada função.
	4.	Somar os deltas e aplicá-los ao modelo base, criando um modelo final que agrega o aprendizado de todas as funções.

### Combinação Otimizada por Mínimos Quadrados

Aplicamos o método dos mínimos quadrados para encontrar a combinação ótima de funções de perda. A ideia é ajustar pesos a₁, a₂, ..., aₙ que combinem as funções f₁(x), f₂(x), ..., fₙ(x) para melhor se aproximarem de uma métrica de referência q(x) (como MSE ou IOA). Isso permite identificar uma combinação mais eficaz do que o uso individual de qualquer função gerada.

As tabelas abaixo registram o desempenho obtido. É notório que, diferentemente da tarefa de classificação, para forecasting, apesar dos avanços em geração automática de funções de perda, os modelos ainda enfrentam desafios para superar aqueles treinados com funções consagradas. Esse cenário evidencia a oportunidade para investigar novas combinações e ajustes finos das funções de perda, de modo a alinhar melhor os critérios de otimização com as métricas de avaliação específicas do forecasting, impulsionando, assim, o desempenho e a robustez dos modelos de previsão.


| Métrica | mlp     | rnn                    |
|---------|---------|------------------------|
| mse     | 1.75E-05| 0.00016                |
| 1    | 0.1725  | 0.0470                 |
| 2    | 0.2188  | 0.0498                 |
| 3    | 0.2369  | 0.0505                 |
| 4    | 0.2386  | 0.0535                 |
| 5    | 0.2414  | 0.0548                 |
| fusão de pesos   | 0.013   | 0.011   |


as melhores 5 funções encontradas para cada modelo são as seguintes: 
### 📘 Funções de Loss – MLP

1. ` (ŷ * (1 / ŷ + y |))²`
2. `(log(ŷ))² * |ŷ * y|`
3. `exp(sqrt(|ŷ²|)) + 1 / log(sqrt(ŷ * y))`
4. `(log(ŷ * y) + exp(tanh(ŷ)))²`
5. `|log(ŷ + y)|`

---

### 📗 Funções de Loss – RNN

1. `(ŷ² * (tanh(ŷ) + ŷ * y))²`
2. `(tanh(ŷ * y) + log(1 / ŷ))²`
3. `sqrt(|ŷ + y|)`
4. `1 / |tanh(ŷ * y)|`
5. `sqrt(tanh((sqrt(ŷ * y))²))`

Em todos os casos, as restrições de domínio das funções são superadas por meio da introdução de constantes pequenas.


### reprodutibilidade 
Todos os experimentos são passíveis de reprodução. Recomenda-se a criação de um ambiente conforme as diretrizes descritas no arquivo de requisitos (requirements.txt).

