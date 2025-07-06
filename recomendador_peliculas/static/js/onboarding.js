// onboarding.js - Código completo
console.log("🚀 Onboarding JS cargado");

// Variables globales
let ratingsCount = 0;
let totalMovies = 0;
const minRatings = 5;
let progressBar, progressText, submitBtn, refreshBtn;

document.addEventListener('DOMContentLoaded', function() {
    console.log("🔍 DOM ready - iniciando onboarding");
    
    // Esperar un poco más por si acaso
    setTimeout(function() {
        console.log("🔍 Timeout ejecutado - buscando elementos");
        
        const ratingContainers = document.querySelectorAll('.rating-container');
        progressBar = document.getElementById('progressBar');
        progressText = document.getElementById('progressText');
        submitBtn = document.getElementById('submitBtn');
        refreshBtn = document.getElementById('refreshBtn');
        
        console.log("🔍 Rating containers encontrados:", ratingContainers.length);
        console.log("🔍 Progress bar:", progressBar);
        console.log("🔍 Progress text:", progressText);
        console.log("🔍 Submit button:", submitBtn);
        console.log("🔍 Refresh button:", refreshBtn);
        
        if (ratingContainers.length === 0) {
            console.error("❌ No se encontraron rating containers");
            return;
        }
        
        if (!progressBar || !progressText || !submitBtn || !refreshBtn) {
            console.error("❌ Faltan elementos esenciales");
            return;
        }
        
        totalMovies = ratingContainers.length;
        console.log("✅ Todos los elementos encontrados, configurando listeners");

        // Configurar event listeners para cada película
        ratingContainers.forEach((container, containerIndex) => {
            const movieId = container.dataset.movieId;
            const stars = container.querySelectorAll('.rating-star');
            const ratingInput = document.querySelector(`input[name="rating_${movieId}"]`);
            const ratingText = container.querySelector('.rating-text small');
            const movieCard = container.closest('.movie-card');

            console.log(`📋 Configurando película ${containerIndex + 1}:`, {
                movieId,
                stars: stars.length,
                ratingInput: !!ratingInput,
                ratingText: !!ratingText,
                movieCard: !!movieCard
            });

            if (!ratingInput || !ratingText || !movieCard) {
                console.error(`❌ Faltan elementos para película ${movieId}`);
                return;
            }

            stars.forEach((star, index) => {
                // Hover effect
                star.addEventListener('mouseenter', () => {
                    highlightStars(stars, index + 1);
                });

                // Click effect
                star.addEventListener('click', () => {
                    console.log(`⭐ Click en estrella ${index + 1} de película ${movieId}`);
                    const rating = index + 1;
                    selectRating(stars, rating, ratingInput, ratingText, movieCard);
                    updateProgress();
                });
            });

            // Reset on mouse leave
            container.addEventListener('mouseleave', () => {
                const currentRating = parseInt(ratingInput.value) || 0;
                highlightStars(stars, currentRating);
            });
        });

        // Form submission con modal de procesamiento
        const form = document.getElementById('onboardingForm');
        if (form) {
            form.addEventListener('submit', function(e) {
                if (ratingsCount < minRatings) {
                    e.preventDefault();
                    alert(`Por favor, califica al menos ${minRatings} películas para continuar.`);
                    return;
                }

                console.log("📤 Enviando formulario...");
                
                // Prevenir envío múltiple
                e.preventDefault();
                
                // Mostrar modal de procesamiento
                const processingModal = new bootstrap.Modal(document.getElementById('processingModal'));
                processingModal.show();
                
                // Deshabilitar botón
                submitBtn.disabled = true;
                
                // Enviar formulario después de mostrar la modal
                setTimeout(() => {
                    console.log("🚀 Enviando formulario realmente...");
                    form.submit();
                }, 500);
            });
        }

        // Initialize progress
        updateProgress();
        
        // Cargar calificaciones existentes si las hay
        cargarCalificacionesExistentes();
        
        // Configurar botón de refrescar
        configurarBotonRefrescar();
        
        console.log("🎉 Onboarding completamente inicializado");
        
    }, 500); // Esperar 500ms para asegurar que todo esté cargado
});

function highlightStars(stars, count) {
    stars.forEach((star, index) => {
        if (index < count) {
            star.classList.add('active');
        } else {
            star.classList.remove('active');
        }
    });
}

function selectRating(stars, rating, input, textElement, card) {
    console.log(`📝 Seleccionando rating ${rating}`);
    
    const wasRated = input.value !== '';
    
    input.value = rating;
    highlightStars(stars, rating);
    
    // Update text
    const ratingTexts = ['', 'Muy mala', 'Mala', 'Regular', 'Buena', 'Excelente'];
    textElement.textContent = `${rating}/5 - ${ratingTexts[rating]}`;
    textElement.className = 'text-success';
    
    // Mark card as rated
    card.classList.add('rated');
    
    // Update counter if it's a new rating
    if (!wasRated) {
        ratingsCount++;
        console.log(`📊 Películas calificadas: ${ratingsCount}/${totalMovies}`);
    }
}

function updateProgress() {
    const percentage = Math.min((ratingsCount / totalMovies) * 100, 100);
    progressBar.style.width = percentage + '%';
    
    progressText.textContent = `${ratingsCount} de ${totalMovies} películas calificadas`;
    
    if (ratingsCount >= minRatings) {
        progressText.textContent += ' ✓ ¡Listo para recomendaciones!';
        submitBtn.disabled = false;
        submitBtn.style.opacity = '1';
        console.log("✅ Mínimo de calificaciones alcanzado");
    } else {
        const remaining = minRatings - ratingsCount;
        progressText.textContent += ` (${remaining} más para continuar)`;
        submitBtn.disabled = true;
        submitBtn.style.opacity = '0.6';
    }
}

function cargarCalificacionesExistentes() {
    // Buscar inputs con valores pre-cargados y aplicar las calificaciones
    document.querySelectorAll('.rating-input').forEach(input => {
        if (input.value && input.value !== '') {
            const movieId = input.name.replace('rating_', '');
            const rating = parseInt(input.value);
            const container = document.querySelector(`[data-movie-id="${movieId}"]`);
            
            if (container) {
                const stars = container.querySelectorAll('.rating-star');
                const ratingText = container.querySelector('.rating-text small');
                const movieCard = container.closest('.movie-card');
                
                // Aplicar la calificación visualmente
                highlightStars(stars, rating);
                
                const ratingTexts = ['', 'Muy mala', 'Mala', 'Regular', 'Buena', 'Excelente'];
                ratingText.textContent = `${rating}/5 - ${ratingTexts[rating]}`;
                ratingText.className = 'text-success';
                movieCard.classList.add('rated');
                
                // Incrementar contador
                ratingsCount++;
            }
        }
    });
    
    console.log(`📊 Calificaciones existentes cargadas: ${ratingsCount}`);
    // Actualizar progreso después de cargar calificaciones existentes
    updateProgress();
}

function configurarBotonRefrescar() {
    refreshBtn.addEventListener('click', function() {
        console.log("🔄 Refrescando películas...");
        
        // Mostrar estado de carga
        refreshBtn.disabled = true;
        refreshBtn.innerHTML = '<i class="spinner-border spinner-border-sm me-2"></i>Cargando nuevas películas...';
        
        // Crear formulario para enviar datos
        const form = document.getElementById('onboardingForm');
        
        // Cambiar la acción del formulario temporalmente
        const originalAction = form.action;
        form.action = '/recomendador-peliculas/refrescar_peliculas';
        
        // Enviar formulario (esto causará una redirección)
        form.submit();
    });
}