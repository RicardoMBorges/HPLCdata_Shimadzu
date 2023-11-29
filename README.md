# HPLCdata_Shimadzu
Este foi criado para extrair dados de arquivos exportados do HPLC-DAD (Shimadzu) do IPPN-UFRJ. 
Ele cria uma tabela com eixo RT(min) e os dados de intensidade de um comprimento de onda; este foi excolhido na etapa de extração do instrumento. 

Início no Instrumento:
1. Selecionar o dado
2. Observar o comprimento de onda usado
3. Observar se há zoom, os tempos de retenção
4. File -> Exportar -> ASCII
5. Selecionar a opção "Chromatogram", apenas esta e desmarcar todas as outras
6. Definir corretamente o caminho onde o novo arquivo será salvo e utilizar o nome da própria amostra para o filename
