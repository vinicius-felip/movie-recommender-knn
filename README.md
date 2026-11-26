# MovieLens 100K — recomendador k-NN
Projeto didático de IA. Python 3.10+ recomendado. Execute na raiz do projeto.

## Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe experiments.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Usar o executável diretamente evita problemas de política de ativação do PowerShell. Em Linux/macOS, use `.venv/bin/python`.
A base não está incluída; o download usa a fonte oficial https://grouplens.org/datasets/movielens/100k/. Também é possível extrair manualmente o ZIP oficial em data/ml-100k/.

## Ordem de estudo
1. src/dataset.py: leitura e matriz de notas (NaN = ausente).
2. src/metrics.py: cinco métricas implementadas com NumPy. Distâncias euclidiana e Manhattan são brutas, Hamming é a proporção de diferenças; peso de distância = 1/(1+d). Cosseno e Pearson usam apenas valores positivos. Mínimo de dois filmes em comum. Pearson com dois pontos pode ser instável.
3. src/knn.py: seleção dos vizinhos que avaliaram cada filme e média ponderada. Sem vizinhos, a previsão experimental usa média do filme ou média global. A interface exclui recomendações sem vizinhos.
4. experiments.py: divisão aleatória por avaliação 64/16/20 mil, semente 42; 30 configurações na validação. Escolha pelo menor RMSE (MAE desempata), reconstrução com treino+validação e teste final contra média do filme. MAE/RMSE incluem fallback; cobertura mede previsões com vizinhos. Esse protocolo testa notas ocultas de usuários conhecidos, não simula especificamente novos usuários com duas notas. Distâncias brutas podem depender do número de filmes em comum; cosseno bruto pode superestimar similaridade. Não há garantia de que o vencedor supere o baseline.
5. app.py: interface local com as mesmas funções; usa a base completa para demonstração.

## Resultados
A execução dos experimentos gera CSVs de validação, teste, previsões e gráficos em results/. A implementação prioriza legibilidade, portanto a execução completa pode ser demorada; execute antes do seminário. Não há resultados experimentais pré-preenchidos.

## Versão 2 — atributos adicionais
- `tradicional`: reproduz a versão 1 (mesma métrica, pesos e divisão).
- `generos`: combina peso de proximidade das notas (50) e perfil por gênero (35).
- `perfil`: acrescenta idade, ocupação e sexo registrado (15). Campos opcionais na interface; não utiliza CEP ou timestamp. Não pressupõe que demografia determine gosto; sua contribuição deve ser testada.
- Perfil por gênero: média das notas conhecidas dos filmes classificados naquele gênero. Filmes multigênero contribuem para cada gênero correspondente. Categoria `unknown` excluída. Similaridade = 1 - média das diferenças absolutas / 4, nos gêneros em comum. Não é a métrica selecionada no menu: ela se aplica ao bloco de notas, mantendo os outros blocos fixos para comparação.
- Demografia: idade 1 - min(diferença/60,1); ocupação e sexo, igualdade (0 ou 1); média dos campos disponíveis. Sem evidência positiva de gosto, não há vizinho exclusivamente demográfico.
- Similaridade final: média ponderada dos blocos disponíveis; se notas em comum forem insuficientes, o bloco de notas é omitido. Uma similaridade válida igual a zero não é tratada como ausente. Os pesos são escolhas iniciais, não otimizados. Para evitar sobreinterpretação, um gênero em comum já permite comparação, mas pode fornecer evidência fraca.
- A previsão continua sendo média ponderada das notas reais dos vizinhos para cada filme; o mínimo de suporte (padrão 5, limitado a k) é aplicado apenas ao ranking. Se não houver filmes suficientes, reduza o suporte ou avalie mais filmes. Não inventamos avaliações ausentes.
- Experimentos: 90 combinações (3 modelos × 5 métricas × 6 valores de k), mesma divisão e semente da V1. Escolhe uma configuração por modelo na validação e avalia cada uma no teste. Os pesos ficam fixos; não são selecionados pelo teste. Resultados em `results_v2/`, sem sobrescrever os anteriores. Compare `validation_winners.csv`, `test_metrics.csv` e gráficos por modelo. Como o teste anterior já foi observado, trate comparações novas como exploratórias; confirmação exigiria novo conjunto independente. O teste mede previsão de notas de usuários conhecidos, não qualidade Top-5 nem cold start específico.
- Interface: use `python -m streamlit run app.py`. Escolha modelo, métrica de notas, k, suporte, filmes e perfil opcional. A demonstração usa a base completa. O ranking geral de vizinhos pode diferir dos vizinhos usados para cada filme.
- Testes: `python -m unittest discover -s tests -v`.

## Modo seminário
Execute `python -m streamlit run seminar.py` na raiz. A apresentação tem 14 páginas com navegação por botões, setas e seleção direta. A página 6 usa as funções reais de `src/metrics.py` e exibe fórmulas, código e resultados. Os resultados são lidos de `results_v2/` ou do ZIP local. A página 13 reutiliza `render_recommender()` de `app.py`, que continua executável separadamente. Use F11 para tela cheia.
