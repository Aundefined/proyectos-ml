document.addEventListener('DOMContentLoaded', function() {
    // Inicializar highlight.js para resaltar el código
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightBlock(block);
    });
    
    // Activar tooltips de Bootstrap
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });
    
    // Ajustar altura del modal de galería según el contenido
    const galeriaModal = document.getElementById('galeriaPrincipal');
    if (galeriaModal) {
        galeriaModal.addEventListener('shown.bs.modal', function () {
            const activeTab = document.querySelector('.tab-pane.active');
            const tabContent = document.getElementById('galeriaTabContent');
            if (activeTab && tabContent) {
                const minHeight = Math.max(400, activeTab.offsetHeight);
                tabContent.style.minHeight = minHeight + 'px';
            }
        });
    }

    // Cambiar altura del contenido de tabs cuando se cambia de tab
    const tabEls = document.querySelectorAll('button[data-bs-toggle="tab"]')
    tabEls.forEach(tabEl => {
        tabEl.addEventListener('shown.bs.tab', event => {
            const activeTab = document.querySelector('.tab-pane.active');
            const tabContent = document.getElementById('galeriaTabContent');
            if (activeTab && tabContent) {
                const minHeight = Math.max(400, activeTab.offsetHeight);
                tabContent.style.minHeight = minHeight + 'px';
            }
        });
    });
});