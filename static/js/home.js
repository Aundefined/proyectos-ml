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
            
            // Asegurarse de que el botón cierre el modal
            document.querySelector('#welcomeModal .btn-primary').onclick = function() {
                modal.hide();
            };
            
            document.querySelector('#welcomeModal .btn-close').onclick = function() {
                modal.hide();
            };
        }, 300);
    }
    
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