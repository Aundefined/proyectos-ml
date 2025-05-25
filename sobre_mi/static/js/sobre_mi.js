// ARCHIVO: static/js/sobre_mi.js

document.addEventListener('DOMContentLoaded', function() {
    // Variables del libro
    let currentPage = 1;
    const pages = document.querySelectorAll('.page');
    const totalPages = pages.length;
    
    // Elementos de control
    const prevBtn = document.getElementById('prevPage');
    const nextBtn = document.getElementById('nextPage');
    const currentPageNum = document.getElementById('currentPageNum');
    const totalPagesNum = document.getElementById('totalPages');
    
    // Inicializar
    totalPagesNum.textContent = totalPages;
    updatePageDisplay();
    
    // Event listeners para los botones
    prevBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            goToPage(currentPage - 1);
        }
    });
    
    nextBtn.addEventListener('click', () => {
        if (currentPage < totalPages) {
            goToPage(currentPage + 1);
        }
    });
    
    // Navegación con teclado
    document.addEventListener('keydown', function(e) {
        if (e.key === 'ArrowLeft' && currentPage > 1) {
            goToPage(currentPage - 1);
        } else if (e.key === 'ArrowRight' && currentPage < totalPages) {
            goToPage(currentPage + 1);
        }
    });
    
    // Función para ir a una página específica
    function goToPage(pageNum) {
        if (pageNum < 1 || pageNum > totalPages) return;
        
        // Quitar clase active de la página actual
        pages[currentPage - 1].classList.remove('active');
        pages[currentPage - 1].classList.add('prev');
        
        // Pequeño delay para la animación
        setTimeout(() => {
            pages[currentPage - 1].classList.remove('prev');
            
            // Activar nueva página
            currentPage = pageNum;
            pages[currentPage - 1].classList.add('active');
            
            updatePageDisplay();
        }, 100);
    }
    
    // Actualizar display de controles
    function updatePageDisplay() {
        currentPageNum.textContent = currentPage;
        
        // Habilitar/deshabilitar botones
        prevBtn.disabled = currentPage === 1;
        nextBtn.disabled = currentPage === totalPages;
        
        // Efecto visual en los botones
        if (currentPage === 1) {
            prevBtn.style.opacity = '0.5';
        } else {
            prevBtn.style.opacity = '1';
        }
        
        if (currentPage === totalPages) {
            nextBtn.style.opacity = '0.5';
        } else {
            nextBtn.style.opacity = '1';
        }
    }
    
    // Efecto de hover en las tarjetas tech
    const techTags = document.querySelectorAll('.tech-tag');
    techTags.forEach(tag => {
        tag.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px) scale(1.05)';
        });
        
        tag.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
    
    // Animación de entrada para elementos
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observar elementos con animación de caída
    document.querySelectorAll('.fall-element').forEach(el => {
        observer.observe(el);
    });
    
    // Efecto parallax sutil en la foto de perfil
    const profilePhoto = document.querySelector('.profile-photo');
    if (profilePhoto) {
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const rate = scrolled * -0.5;
            profilePhoto.style.transform = `translateY(${rate}px) scale(1)`;
        });
    }
    
    // Animación de typing para el título principal
    const mainTitle = document.querySelector('.display-5');
    if (mainTitle) {
        const text = mainTitle.textContent;
        mainTitle.textContent = '';
        let i = 0;
        
        function typeWriter() {
            if (i < text.length) {
                mainTitle.textContent += text.charAt(i);
                i++;
                setTimeout(typeWriter, 50);
            }
        }
        
        // Iniciar animación después de un pequeño delay
        setTimeout(typeWriter, 500);
    }
    
    // Efecto de brillo en el libro al hacer hover
    const book = document.querySelector('.book');
    book.addEventListener('mouseenter', function() {
        this.style.filter = 'drop-shadow(0 15px 35px rgba(123, 31, 162, 0.3))';
    });
    
    book.addEventListener('mouseleave', function() {
        this.style.filter = 'none';
    });
    
    // Función para añadir páginas dinámicamente (para uso futuro)
    window.addNewPage = function(pageData) {
        const pageNumber = pages.length + 1;
        const newPage = document.createElement('div');
        newPage.className = 'page';
        newPage.setAttribute('data-page', pageNumber);
        
        newPage.innerHTML = `
            <div class="page-content">
                <div class="page-header">
                    <h3>${pageData.period}</h3>
                    <div class="company-badge">${pageData.company}</div>
                </div>
                <div class="role-title">${pageData.role}</div>
                <div class="page-body">
                    <p><strong>${pageData.sectionTitle}:</strong></p>
                    <ul>
                        ${pageData.responsibilities.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                    ${pageData.highlight ? `
                        <div class="achievement-highlight">
                            <i class="bi bi-award-fill"></i>
                            <span>${pageData.highlight}</span>
                        </div>
                    ` : ''}
                    <div class="tech-stack">
                        ${pageData.technologies.map(tech => `<span class="tech-tag">${tech}</span>`).join('')}
                    </div>
                </div>
            </div>
            <div class="page-number">${pageNumber}</div>
        `;
        
        document.getElementById('bookPages').appendChild(newPage);
        
        // Actualizar contador
        const newTotalPages = document.querySelectorAll('.page').length;
        document.getElementById('totalPages').textContent = newTotalPages;
        
        return newPage;
    };
});

// Ejemplo de cómo usar la función addNewPage:
/*
const nuevaPagina = {
    period: "2023 - Actualidad", 
    company: "Mi Nueva Empresa",
    role: "Senior Developer",
    sectionTitle: "Principales logros",
    responsibilities: [
        "Desarrollo de nuevas funcionalidades",
        "Liderazgo de equipo",
        "Arquitectura de software"
    ],
    highlight: "Proyecto del año 2023",
    technologies: ["React", "Node.js", "MongoDB", "Docker"]
};

addNewPage(nuevaPagina);
*/