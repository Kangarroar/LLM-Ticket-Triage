import requests
import json
import concurrent.futures
from utils.tokenrefresher import verificar_token
from utils.jsonformatter import convert_jsonl_to_csv
import os
import xml.etree.ElementTree as ET
# --------- Configuración Inicial ----------
# Same repurposed from old project, may modify later and translate
def load_config():
    tree = ET.parse('config.xml')
    root = tree.getroot()
    data_folder = root.find('data_folder').text
    workers1 = int(root.find('workers1').text)
    workers2 = int(root.find('workers2').text)
    return data_folder, workers1, workers2

# Cargar configuración
data_folder, workers1, workers2 = load_config()


row_count = 100  # Cada worker tomará 1 ticket

jsonl_file = os.path.join(data_folder, "tickets.jsonl")
detailed_jsonl_file = os.path.join(data_folder, "tickets_full.jsonl")
csv_file = os.path.join(data_folder, "tickets_full.csv")

# --------- Inicialización del Token ----------
# Asegurarse de que el token esté actualizado antes de empezar
#verificar_token()

# --------- Función para Obtener el Token ----------
def get_token():
    with open("token.json", "r") as token_file:
        data = json.load(token_file)
        return data["access_token"]

# --------- Función para Obtener Tickets ----------
def get_tickets(start_index, row_count):
    url = "https://sdpondemand.manageengine.com/api/v3/requests"
    access_token = get_token()
    headers = {
        "Accept": "application/vnd/manageengine.sdp.v3+json",
        "Authorization": f"Zoho-oauthtoken {access_token}",
    }

    # Ahora configuramos row_count a 100
    params = {
        "input_data": json.dumps({
            "list_info": {"row_count": row_count, "page": start_index}
        })
    }

    response = requests.get(url, headers=headers, params=params, verify=True)
    if response.status_code == 200:
        return response.json().get("requests", [])
    else:
        return []

# --------- Función para Extraer Todos los Tickets en Paralelo ----------
def fetch_all_tickets_parallel(row_count):
    all_tickets = []
    start_index = 1
    tickets_remaining = True
    print("Iniciando la obtención de tickets...")

    # Usamos ThreadPoolExecutor para crear 5 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers1) as executor:
        futures = []
        while tickets_remaining:
            # Enviamos solicitudes en paralelo, cada una con row_count=100
            for worker_id in range(workers1):
                current_start_index = start_index + worker_id  # A cada worker le asignamos un índice diferente
                futures.append(executor.submit(get_tickets, current_start_index, row_count))
                print(f"Worker {worker_id + 1} solicitando desde el índice {current_start_index}...")

            # Esperamos a que todas las solicitudes se completen
            for future in concurrent.futures.as_completed(futures):
                tickets = future.result()
                if len(tickets) < row_count:
                    tickets_remaining = False  # Si algún worker obtiene menos de 100 tickets, se detiene el proceso
                if tickets:
                    print(f"Se obtuvieron {len(tickets)} tickets.")
                    all_tickets.extend(tickets)
            
            # Incrementamos el índice de inicio para la siguiente ronda de solicitudes
            start_index += workers1
            futures.clear()  # Limpiamos las tareas anteriores

    print(f"\nSe extrajeron {len(all_tickets)} tickets en total.")
    return all_tickets

# --------- Función para Guardar Tickets en JSONL ----------
def save_to_jsonl(tickets, jsonl_file):
    with open(jsonl_file, mode="w", encoding="utf-8") as file:
        for ticket in tickets:
            file.write(json.dumps(ticket) + "\n")
    print(f"Se guardaron {len(tickets)} tickets en el archivo {jsonl_file}.")

# --------- Función para Obtener Detalles de un Lote de Tickets ----------
def fetch_ticket_details_batch(ticket_ids, all_details):
    # Aquí procesamos cada ticket individualmente
    for ticket_id in ticket_ids:
        ticket_details = get_ticket_details(ticket_id)
        if ticket_details:
            all_details.append(ticket_details)

# --------- Función para Obtener Detalles de un Ticket Específico ----------
def get_ticket_details(ticket_id):
    url = f"https://sdpondemand.manageengine.com/api/v3/requests/{ticket_id}"
    access_token = get_token()
    headers = {
        "Accept": "application/vnd/manageengine.sdp.v3+json",
        "Authorization": f"Zoho-oauthtoken {access_token}",
    }

    response = requests.get(url, headers=headers, verify=True)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error obteniendo el ticket {ticket_id}: {response.status_code}")
        return None

# --------- Función para Obtener Todos los Detalles de Tickets ----------
def fetch_ticket_details_from_jsonl(jsonl_file, detailed_jsonl_file):
    with open(jsonl_file, "r", encoding="utf-8") as file:
        ticket_ids = [json.loads(line)["id"] for line in file]
    
    all_details = []
    completed_tickets = 0  # Variable para contar los tickets completados
    print("Empezando ejecución detallada...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers2) as executor:
        futures = []
        start = 0
        while start < len(ticket_ids):
            # Asignar un solo ticket a cada worker
            batch_ids = ticket_ids[start:start + 1]  # Un ticket por worker
            futures.append(executor.submit(fetch_ticket_details_batch, batch_ids, all_details))
            start += 1  # Incrementar de uno en uno

        # Esperar que todos los hilos finalicen
        concurrent.futures.wait(futures)

        # Imprimir después de que cada 100 tickets hayan sido procesados
        for i in range(100, len(all_details), 100):
            print(f"Se han ejecutado {i} tickets.")
        
        # Si hay menos de 100 tickets restantes, imprimir un mensaje final con el total
        if len(all_details) % 100 != 0:
            print(f"Se han ejecutado {len(all_details)} tickets.")

    # Guardar los detalles completos de los tickets en un nuevo archivo JSONL
    with open(detailed_jsonl_file, mode="w", encoding="utf-8") as file:
        json.dump(all_details, file, ensure_ascii=False, indent=4)
    print(f"Se guardaron los detalles completos de los tickets en el archivo {detailed_jsonl_file}.")


    
# --------- Flujo Principal ----------
# Obtener todos los tickets en paralelo
all_tickets = fetch_all_tickets_parallel(row_count)

# Guardar los tickets en formato JSON Lines
save_to_jsonl(all_tickets, jsonl_file)

# Obtener los detalles de cada ticket y guardarlos en otro JSONL
fetch_ticket_details_from_jsonl(jsonl_file, detailed_jsonl_file)

# Convertir el archivo JSON Lines con detalles completos a CSV
convert_jsonl_to_csv(detailed_jsonl_file, csv_file)
