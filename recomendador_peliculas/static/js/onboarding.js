// onboarding.js - Código completo
console.log("🚀 Onboarding JS cargado");

document.addEventListener('DOMContentLoaded', function() {
    console.log("🔍 DOM ready - iniciando onboarding");
    
    // Esperar un poco más por si acaso
    setTimeout(function() {
        console.log("🔍 Timeout ejecutado - buscando elementos");
        
        const ratingContainers = document.querySelectorAll('.rating-container');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const submitBtn = document.getElementById('submitBtn');
        
        console.log("🔍 Rating containers encontrados:", ratingContainers.length);
        console.log("🔍 Progress bar:", progressBar);
        console.log("🔍 Progress text:", progressText);
        console.log("🔍 Submit button:", submitBtn);
        
        if (ratingContainers.length === 0) {
            console.error("❌ No se encontraron rating containers");
            return;
        }
        
        if (!progressBar || !progressText || !submitBtn) {
            console.error("❌ Faltan elementos esenciales");
            return;
        }
        
        let ratingsCount = 0;
        const totalMovies = ratingContainers.length;
        const minRatings = 5;

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

        // Auto-scroll after rating
        function scrollToNextUnrated() {
            const unratedCard = document.querySelector('.movie-card:not(.rated)');
            if (unratedCard) {
                unratedCard.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'center' 
                });
            }
        }

        // Form submission with loading state
        const form = document.getElementById('onboardingForm');
        if (form) {
            form.addEventListener('submit', function(e) {
                if (ratingsCount < minRatings) {
                    e.preventDefault();
                    alert(`Por favor, califica al menos ${minRatings} películas para continuar.`);
                    return;
                }

                console.log("📤 Enviando formulario...");
                
                // Show loading state
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="spinner-border spinner-border-sm me-2"></i>Generando recomendaciones...';
                
                // Allow form to submit
                setTimeout(() => {
                    submitBtn.innerHTML = '<i class="bi bi-magic me-2"></i>Procesando...';
                }, 1000);
            });
        }

        // Initialize progress
        updateProgress();
        
        console.log("🎉 Onboarding completamente inicializado");
        
    }, 500); // Esperar 500ms para asegurar que todo esté cargado
});