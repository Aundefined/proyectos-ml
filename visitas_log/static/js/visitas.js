$(document).ready(function() {
    $('#visitasTable').DataTable({
        "order": [[ 0, "desc" ]], // Ordenar por fecha descendente por defecto
        "pageLength": 25,
        "lengthMenu": [[10, 25, 50, 100, -1], [10, 25, 50, 100, "Todos"]],
        "responsive": true,
        "language": {
            "url": "//cdn.datatables.net/plug-ins/1.13.6/i18n/es-ES.json"
        },
        "columnDefs": [
            {
                "targets": 0,
                "width": "10%"
            },
            {
                "targets": 1,
                "width": "18%"
            },
            {
                "targets": 2,
                "width": "10%"
            },
            {
                "targets": 3,
                "width": "8%"
            },
            {
                "targets": 4,
                "width": "10%"
            },
            {
                "targets": 5,
                "width": "10%"
            },
            {
                "targets": 6,
                "width": "10%"
            },
            {
                "targets": 7,
                "width": "6%"
            },
            {
                "targets": 8,
                "width": "8%",
                "orderable": false
            },
            {
                "targets": 9,
                "width": "10%",
                "orderable": false
            }
        ]
    });
    
    // Manejar eliminación de registros
    $(document).on('click', '.delete-visit', function() {
        var visitId = $(this).data('visit-id');
        var row = $(this).closest('tr');
        
        if (confirm('¿Estás seguro de que quieres eliminar este registro de visita?')) {
            $.ajax({
                url: '/visitas/delete/' + visitId,
                type: 'POST',
                success: function(response) {
                    if (response.success) {
                        // Eliminar la fila de la tabla
                        $('#visitasTable').DataTable().row(row).remove().draw();
                        
                        // Mostrar mensaje de éxito
                        $('<div class="alert alert-success alert-dismissible fade show" role="alert">' +
                          response.message +
                          '<button type="button" class="btn-close" data-bs-dismiss="alert"></button>' +
                          '</div>').prependTo('.card-body').delay(3000).fadeOut();
                    } else {
                        alert('Error: ' + response.message);
                    }
                },
                error: function() {
                    alert('Error al comunicarse con el servidor');
                }
            });
        }
    });
});