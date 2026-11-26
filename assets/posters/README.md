# Pôsteres da página 2

Execute `python download_posters.py` na raiz do projeto uma vez, com internet.
O script salva as imagens em JPEG local. A apresentação não acessa sites externos.
Se uma fonte estiver indisponível, coloque o pôster correspondente manualmente aqui.

| Arquivo | Fonte do pôster promocional |
|---|---|
| `star_wars.jpg` | [Original Film Art — Star Wars (1977)](https://www.originalfilmart.com/products/star-wars-a) |
| `terminator_2.jpg` | [Film Art Gallery — Terminator 2 (1991)](https://filmartgallery.com/products/terminator-2) |
| `titanic.jpg` | [Original Film Art — Titanic (1997)](https://www.originalfilmart.com/products/titanic-ds-a) |
| `pretty_woman.jpg` | [Film Art Gallery — Pretty Woman (1990)](https://filmartgallery.com/products/pretty-woman-6) |

Pôsteres promocionais de terceiros, para ilustração acadêmica. Os direitos pertencem aos respectivos titulares.

## Catálogo offline — patch 014

Antes do seminário, execute `python download_catalog_posters.py` com internet.
Ele importa a coleção pública [mohanvpatta/movie-posters](https://github.com/mohanvpatta/movie-posters),
identificada pelo ID MovieLens 100K, em `assets/posters/catalog/<movie_id>.jpg`.
Também reaproveita as quatro capas da página 2, caso já existam.
Não é necessário token, cadastro, ou acesso remoto durante a execução do app.

A coleção externa não contém necessariamente todos os 1.682 filmes e pode ter
imagens incorretas ou ausentes. Confira o resultado antes da apresentação.
Para substituir ou completar uma capa, salve o JPEG com o ID correto na pasta
`catalog/`. Para verificar o cache sem internet:

```powershell
python download_catalog_posters.py --check
```

Alternativa, se já tiver o ZIP da coleção:

```powershell
python download_catalog_posters.py --archive C:\caminho\movie-posters-main.zip
```

Os pôsteres são usados somente na interface, não participam dos cálculos.
A aplicação lê e incorpora exclusivamente arquivos locais. Se algum faltar,
exibe “Sem capa” sem interromper o recomendador. As imagens permanecem
propriedade dos titulares; utilize para a apresentação acadêmica e não redistribua.

Se ainda faltarem imagens, uma tentativa adicional (com internet) é:

```powershell
python download_catalog_posters.py --fill-missing
```

Essa opção consulta somente os IDs sem capa na lista pública
[babu-thomas/movielens-posters](https://github.com/babu-thomas/movielens-posters).
Links antigos podem estar indisponíveis; o relatório informa a cobertura real.
