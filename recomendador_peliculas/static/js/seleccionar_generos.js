document.addEventListener('DOMContentLoaded', function() {
    const checkboxes = document.querySelectorAll('.genre-checkbox');
    const selectedCount = document.getElementById('selectedCount');
    const continueBtn = document.getElementById('continueBtn');
    const selectAllBtn = document.getElementById('selectAllBtn');
    const clearAllBtn = document.getElementById('clearAllBtn');
    const infoText = document.getElementById('infoText');

    function updateUI() {
        const selected = document.querySelectorAll('.genre-checkbox:checked');
        const count = selected.length;
        
        // Actualizar contador
        selectedCount.textContent = `${count} seleccionado${count !== 1 ? 's' : ''}`;
        
        // Habilitar/deshabilitar botón continuar
        continueBtn.disabled = count < 3;
        
        // Actualizar texto informativo
        if (count === 0) {
            infoText.textContent = 'Selecciona al menos 3 géneros para continuar';
        } else if (count < 3) {
            const remaining = 3 - count;
            infoText.textContent = `Selecciona ${remaining} género${remaining > 1 ? 's' : ''} más para continuar`;
        } else if (count === 3) {
            infoText.textContent = 'Te mostraremos hasta 15 películas (5 por género)';
        } else {
            const peliculas = Math.min(count * 5, 75); // máximo razonable
            infoText.textContent = `Te mostraremos hasta ${peliculas} películas (máximo 5 por género)`;
        }
        
        // Actualizar iconos de las etiquetas
        checkboxes.forEach(checkbox => {
            const label = document.querySelector(`label[for="${checkbox.id}"]`);
            const icon = label.querySelector('i');
            
            if (checkbox.checked) {
                icon.className = 'bi bi-check-square-fill me-2';
            } else {
                icon.className = 'bi bi-square me-2';
            }
        });
    }

    // Event listeners para checkboxes
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateUI);
    });

    // Seleccionar todo
    selectAllBtn.addEventListener('click', function() {
        checkboxes.forEach(checkbox => {
            checkbox.checked = true;
        });
        updateUI();
    });

    // Limpiar todo
    clearAllBtn.addEventListener('click', function() {
        checkboxes.forEach(checkbox => {
            checkbox.checked = false;
        });
        updateUI();
    });

    // Form submission
    document.getElementById('genreForm').addEventListener('submit', function(e) {
        const selected = document.querySelectorAll('.genre-checkbox:checked');
        
        if (selected.length < 3) {
            e.preventDefault();
            alert('Por favor, selecciona al menos 3 géneros');
            return;
        }

        // Mostrar estado de carga
        continueBtn.disabled = true;
        continueBtn.innerHTML = '<i class="spinner-border spinner-border-sm me-2"></i>Cargando películas...';
    });

    // Inicializar UI
    updateUI();
});