// Esperar a que el DOM esté completamente cargado
document.addEventListener('DOMContentLoaded', function() {
    // Verificar si la clave existe en sessionStorage
    if (!sessionStorage.getItem('modalShown')) {
        // Si no existe, mostrar el modal manualmente
        var modalElement = document.getElementById('welcomeModal');
        
        // Crear instancia del modal
        var modal = new bootstrap.Modal(modalElement);
        
        // Mostrar el modal
        setTimeout(function() {
            modal.show();
            
            // Marcar que el modal ya se mostró
            sessionStorage.setItem('modalShown', 'true');
            
            // Asegurarse de que el botón cierre el modal correctamente
            document.querySelector('#welcomeModal .btn-primary').addEventListener('click', function() {
                modal.hide();
                // Eliminar manualmente el backdrop y restaurar el body
                const backdrop = document.querySelector('.modal-backdrop');
                if (backdrop) {
                    backdrop.remove();
                }
                document.body.classList.remove('modal-open');
                document.body.style.overflow = '';
                document.body.style.paddingRight = '';
            });
            
            document.querySelector('#welcomeModal .btn-close').addEventListener('click', function() {
                modal.hide();
                // Eliminar manualmente el backdrop y restaurar el body
                const backdrop = document.querySelector('.modal-backdrop');
                if (backdrop) {
                    backdrop.remove();
                }
                document.body.classList.remove('modal-open');
                document.body.style.overflow = '';
                document.body.style.paddingRight = '';
            });
        }, 300);
    }
    
    // También manejar el evento oculto del modal para limpiar
    document.getElementById('welcomeModal').addEventListener('hidden.bs.modal', function () {
        const backdrop = document.querySelector('.modal-backdrop');
        if (backdrop) {
            backdrop.remove();
        }
        document.body.classList.remove('modal-open');
        document.body.style.overflow = '';
        document.body.style.paddingRight = '';
    });
    
    // Inicializar el carrusel con opciones específicas
    var carousel = new bootstrap.Carousel(document.getElementById('projectsCarousel'), {
        interval: 6000,  // Cambia cada 6 segundos
        wrap: true,     // Permite ciclo continuo
        touch: true,    // Permite control táctil
        pause: 'hover'  // Pausa al poner el cursor encima
    });
    
    // Añadir efecto de hover a las tarjetas
    const cards = document.querySelectorAll('.project-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 20px rgba(0, 0, 0, 0.2)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.1)';
        });
    });
});

// Script para el efecto de fading al cargar la página
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
        document.body.classList.add('loaded');
    }, 20); // Pequeño retraso para asegurar que la transición se active
});