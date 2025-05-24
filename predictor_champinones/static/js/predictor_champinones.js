// Validación del formulario
(function() {
    'use strict';
    
    // Elementos del formulario
    const form = document.querySelector('form');
    const resetBtn = document.getElementById('resetBtn');
    
    // Validación al enviar
    if (form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    }
    
    // Botón de reset
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            if (form) {
                // Resetear el formulario
                form.reset();
                
                // Remover la clase de validación
                form.classList.remove('was-validated');
                
                // Resetear todos los selectores a su opción por defecto
                const selectores = form.querySelectorAll('select');
                selectores.forEach(function(select) {
                    select.selectedIndex = 0; // Selecciona la primera opción (disabled)
                });
                
                // Limpiar el resultado si existe
                const resultadoSection = document.getElementById('resultadoSection');
                if (resultadoSection) {
                    // Opción 1: Ocultar completamente
                    resultadoSection.style.display = 'none';
                    
                    // Opción 2: Si prefieres hacer fade out
                    // resultadoSection.style.opacity = '0';
                    // setTimeout(function() {
                    //     resultadoSection.style.display = 'none';
                    // }, 500);
                }
            }
        });
    }
    
    // Animación de entrada para el resultado
    const resultadoSection = document.getElementById('resultadoSection');
    if (resultadoSection) {
        resultadoSection.style.opacity = '0';
        resultadoSection.style.transform = 'translateY(20px)';
        
        setTimeout(function() {
            resultadoSection.style.transition = 'all 0.5s ease';
            resultadoSection.style.opacity = '1';
            resultadoSection.style.transform = 'translateY(0)';
        }, 100);
    }
    
    // Añadir tooltips a los campos del formulario
    const formControls = document.querySelectorAll('.form-control');
    formControls.forEach(function(control) {
        control.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        control.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
        });
    });
})();

// Animar las barras de progreso
document.addEventListener('DOMContentLoaded', function() {
    const progressComestible = document.getElementById('progress-comestible');
    const progressVenenoso = document.getElementById('progress-venenoso');
    
    if (progressComestible && progressVenenoso) {
        const widthComestible = progressComestible.getAttribute('data-width');
        const widthVenenoso = progressVenenoso.getAttribute('data-width');
        
        // Aplicar los anchos con animación
        setTimeout(function() {
            progressComestible.style.width = widthComestible + '%';
            progressVenenoso.style.width = widthVenenoso + '%';
        }, 100);
    }
});