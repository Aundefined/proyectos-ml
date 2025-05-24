// Validación del formulario y vista previa de imagen
(function() {
    'use strict';
    
    // Elementos del formulario
    const form = document.querySelector('form');
    const resetBtn = document.getElementById('resetBtn');
    const imageInput = document.getElementById('imagen');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('preview');
    
    // Vista previa de imagen
    if (imageInput) {
        imageInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            
            if (file) {
                // Validar tipo de archivo
                const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp'];
                if (!validTypes.includes(file.type)) {
                    alert('Por favor seleccione un archivo de imagen válido (PNG, JPG, JPEG, GIF, BMP)');
                    imageInput.value = '';
                    return;
                }
                
                // Validar tamaño (5MB)
                if (file.size > 5 * 1024 * 1024) {
                    alert('El archivo es demasiado grande. Máximo 5MB permitidos.');
                    imageInput.value = '';
                    return;
                }
                
                // Mostrar vista previa
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImg.src = e.target.result;
                    imagePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            } else {
                imagePreview.style.display = 'none';
            }
        });
    }
    
    // Validación al enviar
    if (form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
            
            // Mostrar indicador de carga
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Procesando...';
                submitBtn.disabled = true;
            }
        }, false);
    }
    
    // Botón de reset
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            if (form) {
                form.reset();
                form.classList.remove('was-validated');
                
                // Ocultar vista previa
                imagePreview.style.display = 'none';
                
                // Limpiar el resultado si existe
                const resultadoSection = document.getElementById('resultadoSection');
                if (resultadoSection) {
                    resultadoSection.style.display = 'none';
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
    
    // Animar las barras de progreso
    document.addEventListener('DOMContentLoaded', function() {
        const progressBars = document.querySelectorAll('.progress-bar');
        
        progressBars.forEach(function(bar) {
            const width = bar.getAttribute('data-width');
            if (width) {
                setTimeout(function() {
                    bar.style.width = width + '%';
                }, 100);
            }
        });
    });
})();