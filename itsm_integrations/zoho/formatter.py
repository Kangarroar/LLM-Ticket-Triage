import json
# Repurposed from old project, may modify later and translate?

def safe_get(data, key, default=""):
    """
    Obtiene de manera segura el valor de una clave en un diccionario,
    devolviendo un valor por defecto si la clave no existe o si el valor es None.
    También maneja casos donde la entrada no es un diccionario o es None.
    """
    if isinstance(data, dict):  # Si es un diccionario, tratamos de obtener la clave
        return data.get(key, default) if data.get(key) is not None else default
    return default  # En cualquier otro caso (None, lista, otro tipo), devolvemos el valor por defecto


def format_tickets(ticket):
    """
    Formatea un ticket JSON para convertirlo a un formato plano adecuado para CSV.
    """
    formatted_ticket = {}

    # Extraer la información dentro de "request"
    request = ticket.get("request", {})

    # Extraer información del requester
    requester = safe_get(request, "requester", {})
    formatted_ticket["requester_email"] = safe_get(requester, "email_id")
    formatted_ticket["requester_is_technician"] = safe_get(requester, "is_technician")
    formatted_ticket["requester_sms_mail"] = safe_get(requester, "sms_mail")
    formatted_ticket["requester_mobile"] = safe_get(requester, "mobile")
    formatted_ticket["requester_last_name"] = safe_get(requester, "last_name")
    formatted_ticket["requester_user_scope"] = safe_get(requester, "user_scope")
    formatted_ticket["requester_sms_mail_id"] = safe_get(requester, "sms_mail_id")
    formatted_ticket["requester_cost_per_hour"] = safe_get(requester, "cost_per_hour")
    formatted_ticket["requester_phone"] = safe_get(requester, "phone")
    formatted_ticket["requester_employee_id"] = safe_get(requester, "employee_id")
    formatted_ticket["requester_name"] = safe_get(requester, "name")
    formatted_ticket["requester_id"] = safe_get(requester, "id")
    formatted_ticket["requester_photo_url"] = safe_get(requester, "photo_url")
    formatted_ticket["requester_is_vip_user"] = safe_get(requester, "is_vip_user")

    # Aseguramos que department y site no sean None
    requester_department = safe_get(requester, "department", {})
    requester_site = safe_get(requester_department, "site", {})
    formatted_ticket["requester_department_name"] = safe_get(requester_department, "name")
    formatted_ticket["requester_department_site_name"] = safe_get(requester_site, "name")
    formatted_ticket["requester_site_name"] = safe_get(requester.get("site", {}), "name")
    formatted_ticket["requester_site_id"] = safe_get(requester.get("site", {}), "id")
    
    # Extraer información del technician
    technician = safe_get(request, "technician", {})
    formatted_ticket["technician_email"] = safe_get(technician, "email_id")
    formatted_ticket["technician_is_technician"] = safe_get(technician, "is_technician")
    formatted_ticket["technician_sms_mail"] = safe_get(technician, "sms_mail")
    formatted_ticket["technician_mobile"] = safe_get(technician, "mobile")
    formatted_ticket["technician_last_name"] = safe_get(technician, "last_name")
    formatted_ticket["technician_user_scope"] = safe_get(technician, "user_scope")
    formatted_ticket["technician_sms_mail_id"] = safe_get(technician, "sms_mail_id")
    formatted_ticket["technician_cost_per_hour"] = safe_get(technician, "cost_per_hour")
    formatted_ticket["technician_phone"] = safe_get(technician, "phone")
    formatted_ticket["technician_employee_id"] = safe_get(technician, "employee_id")
    formatted_ticket["technician_name"] = safe_get(technician, "name")
    formatted_ticket["technician_id"] = safe_get(technician, "id")
    formatted_ticket["technician_photo_url"] = safe_get(technician, "photo_url")
    formatted_ticket["technician_is_vip_user"] = safe_get(technician, "is_vip_user")
    formatted_ticket["technician_department_id"] = safe_get(technician.get("department", {}), "id")
    formatted_ticket["technician_site_id"] = safe_get(technician.get("site", {}), "id")
    
    # Extraer detalles del ticket
    formatted_ticket["ticket_subject"] = safe_get(request, "subject")
    formatted_ticket["ticket_display_id"] = safe_get(request, "display_id")
    formatted_ticket["ticket_display_key"] = safe_get(request.get("display_key", {}), "display_value")
    formatted_ticket["ticket_status_name"] = safe_get(request.get("status", {}), "name")
    formatted_ticket["ticket_status_internal_name"] = safe_get(request.get("status", {}), "internal_name")
    formatted_ticket["ticket_status_color"] = safe_get(request.get("status", {}), "color")
    formatted_ticket["ticket_status_in_progress"] = safe_get(request.get("status", {}), "in_progress")
    formatted_ticket["ticket_status_stop_timer"] = safe_get(request.get("status", {}), "stop_timer")
    formatted_ticket["ticket_creation_time"] = safe_get(request.get("created_time", {}), "display_value")
    formatted_ticket["ticket_due_by_time"] = safe_get(request.get("due_by_time", {}), "display_value")
    formatted_ticket["ticket_is_service_request"] = safe_get(request, "is_service_request")
    formatted_ticket["ticket_cancellation_requested"] = safe_get(request, "cancellation_requested")
    formatted_ticket["ticket_has_notes"] = safe_get(request, "has_notes")
    formatted_ticket["ticket_maintenance"] = safe_get(request, "maintenance")
    formatted_ticket["ticket_editor_status"] = safe_get(request, "editor_status")
    formatted_ticket["ticket_id"] = safe_get(request, "id")
    
    # **Información de resolución**
    resolution = safe_get(request, "resolution", {})
    formatted_ticket["resolution_submitted_on"] = safe_get(resolution, "submitted_on", {}).get("display_value")
    formatted_ticket["resolution_submitted_by_name"] = safe_get(resolution.get("submitted_by", {}), "name")
    formatted_ticket["resolution_submitted_by_email"] = safe_get(resolution.get("submitted_by", {}), "email_id")
    formatted_ticket["resolution_submitted_by_is_technician"] = safe_get(resolution.get("submitted_by", {}), "is_technician")
    formatted_ticket["resolution_submitted_by_phone"] = safe_get(resolution.get("submitted_by", {}), "phone")
    formatted_ticket["resolution_id"] = safe_get(resolution, "id")
    formatted_ticket["resolution_status"] = safe_get(resolution, "status")
    formatted_ticket["resolution_is_trashed"] = safe_get(resolution, "is_trashed")
    formatted_ticket["resolution_lifecycle"] = safe_get(resolution, "lifecycle")

    # **Información SLA**
    sla = safe_get(request, "sla", {})
    formatted_ticket["sla_resolution_due_by_minutes"] = safe_get(sla, "resolution_due_by_minutes")
    formatted_ticket["sla_resolution_due_by_hours"] = safe_get(sla, "resolution_due_by_hours")
    formatted_ticket["sla_resolution_due_by_days"] = safe_get(sla, "resolution_due_by_days")
    formatted_ticket["sla_name"] = safe_get(sla, "name")
    formatted_ticket["sla_is_service_sla"] = safe_get(sla, "is_service_sla")
    formatted_ticket["sla_inactive"] = safe_get(sla, "inactive")

    # Extraer información del group
    group = safe_get(request, "group", {})
    formatted_ticket["group_name"] = safe_get(group, "name")
    
    # Extraer información del item
    item = safe_get(request, "item", {})
    formatted_ticket["item_name"] = safe_get(item, "name")

    # Extraer información sobre el vencimiento
    formatted_ticket["is_overdue"] = safe_get(request, "is_overdue", False)

     # Extraer el nombre de la categoría
    formatted_ticket["service_category"] = safe_get(request.get("service_category", {}), "name")

        # Extraer el nombre de la prioridad
    formatted_ticket["ticket_priority_name"] = safe_get(request.get("priority", {}), "name")

        # Extraer el nombre de la categoría
    formatted_ticket["ticket_category_name"] = safe_get(request.get("category", {}), "name")

    formatted_ticket["ticket_request_name"] = safe_get(request.get("request_type", {}), "name")






    return formatted_ticket 


def process_tickets_file(file_path):
    """
    Procesa un archivo JSON (array de tickets), formateando cada ticket.
    """
    formatted_tickets = []
    
    # Abrimos el archivo y cargamos el JSON array completo
    with open(file_path, 'r', encoding='utf-8') as file:
        tickets = json.load(file)  # Carga el archivo completo como un array JSON

        for ticket in tickets:
            if isinstance(ticket, dict):  # Verificamos que el ticket sea un diccionario
                formatted_tickets.append(format_tickets(ticket))
    
    return formatted_tickets
