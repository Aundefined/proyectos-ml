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