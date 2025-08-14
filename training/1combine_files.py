#COMBINADO que inclui o header do ficheiro csv.
#FINALISSIMO E O QUE ESTÁ A SER USADO DIA 2 DE JUNHO
import csv
from pathlib import Path

MAX_TIMESTAMP_DIF = 200

# Caminhos a alterar em cada teste
folder = "../data/dataset30"
ficheiro_b = "v1.csv" #ficheiro menor

#Caminhos fixos
folder_base = "../data/0base"
ficheiro_a = "dados_teclas1.csv" #ficheiro maior
resultado = "combinado.csv"
csv_path_a = Path(__file__).resolve().parent / folder_base / ficheiro_a #ficheiro maior
csv_path_b = Path(__file__).resolve().parent / folder / ficheiro_b #ficheiro menor
csv_path_output = Path(__file__).resolve().parent / folder / resultado

# Função para carregar dados de um ficheiro CSV
def load_csv_data(path_csv):
    with path_csv.open(mode='r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        rows = []
        for row in reader:
            timestamp = int(row[0])
            other_values = row[1:]
            rows.append([timestamp] + other_values)
    return header, rows


# Carregar os dados dos dois ficheiros
header_a, rows_a = load_csv_data(csv_path_a)
header_b, rows_b = load_csv_data(csv_path_b)

# Garantir que o ficheiro A é o que tem mais linhas
if len(rows_b) > len(rows_a):
    rows_a_copy = list(rows_a)
    rows_a = list(rows_b)
    rows_b = rows_a_copy
    header_a_copy = list(header_a)
    header_a = list(header_b)
    header_b = header_a_copy

# Alinhar inicio de ambos os ficheiros
first_timestamp_dif = abs(rows_a[0][0] - rows_b[0][0])
rows_starting_later = rows_a if rows_a[0][0] > rows_b[0][0] else rows_b
for row in rows_starting_later:
    row[0] -= first_timestamp_dif

# Lista para armazenar as linhas combinadas e as diferenças de tempo
merged_rows = []
timestamp_diffs = []

# Combinar as linhas com base nos timestamps mais próximos
for row_a in rows_a:
    timestamp_a = row_a[0]
    diffs = [abs(timestamp_a - row_b[0]) for row_b in rows_b]
    min_diff = min(diffs)
    if min_diff > MAX_TIMESTAMP_DIF:
        continue  # Ignorar linhas que não têm uma correspondência temporalmente próxima
    timestamp_diffs.append(min_diff)
    closest_index = diffs.index(min_diff)
    row_b = rows_b[closest_index]

    # Combinar: tempo header de A + cabeçalho de A (sem timestamp) + cabeçalho de B (sem timestamp) 
    combined_header = [header_a[0]] + header_a[1:] + header_b[1:]
    # Combinar: timestamp de A + valores de C (sem timestamp) + valores de B (sem timestamp)
    combined_row = [timestamp_a] + row_a[1:] + row_b[1:]
    merged_rows.append(combined_row)

# Escrever o ficheiro resultante
with csv_path_output.open(mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(combined_header)
    writer.writerows(merged_rows)

# Imprimir estatísticas sobre os desvios temporais
max_diff = max(timestamp_diffs)
avg_diff = sum(timestamp_diffs) / len(timestamp_diffs)
print(f'Diferença máxima de timestamp: {max_diff} ms')
print(f'Diferença média de timestamp: {avg_diff:.2f} ms')
