document.addEventListener('DOMContentLoaded', function() {
    // Función para realizar el scroll
    function scrollToResults() {
        const resultadoSection = document.getElementById('resultadoSection');
        if (resultadoSection) {
            // Usar un tiempo de espera más largo para dispositivos móviles
            setTimeout(function() {
                // Método alternativo para hacer scroll
                window.scrollTo({
                    top: resultadoSection.offsetTop - 20, // Un poco de margen en la parte superior
                    behavior: 'smooth'
                });
            }, 800); // Aumentado a 800ms para dar más tiempo en dispositivos móviles
        }
    }

    // Primera intención: al cargar el DOM
    scrollToResults();
    
    // Segunda intención: después de que todas las imágenes y recursos se hayan cargado
    window.addEventListener('load', function() {
        scrollToResults();
    });
    
    // Tercera intención: un tiempo adicional para estar seguros
    setTimeout(scrollToResults, 1500);
});

document.addEventListener('DOMContentLoaded', function() {
    // Scroll al resultado
    function scrollToResults() {
        const resultadoSection = document.getElementById('resultadoSection');
        if (resultadoSection) {
            setTimeout(function() {
                window.scrollTo({
                    top: resultadoSection.offsetTop - 20,
                    behavior: 'smooth'
                });
            }, 800);
        }
    }
    scrollToResults();
    window.addEventListener('load', function() {
        scrollToResults();
    });
    setTimeout(scrollToResults, 1500);

    // --- RESETEAR inputs y selects a vacío ---
    const resetBtn = document.getElementById('resetBtn');
    const form = document.querySelector('form');
    const resultadoSection = document.getElementById('resultadoSection');
    if (resetBtn && form) {
        resetBtn.addEventListener('click', function() {
            // Vacía todos los inputs y selects del formulario
            Array.from(form.elements).forEach(function(el) {
                if (el.tagName === "INPUT" || el.tagName === "SELECT" || el.tagName === "TEXTAREA") {
                    if (el.type !== "submit" && el.type !== "button") {
                        el.value = '';
                        // Los selects deben volver al primer option
                        if (el.tagName === "SELECT") {
                            el.selectedIndex = 0;
                        }
                    }
                }
            });
            // Oculta el resultado si existe
            if (resultadoSection) {
                resultadoSection.style.display = 'none';
            }
        });
    }
});
