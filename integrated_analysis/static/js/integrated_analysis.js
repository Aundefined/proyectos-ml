$(document).ready(function() {
    // Referencias a los iframes
    const visualizacionIframe = document.getElementById('visualizacionIframe');
    const analisisNoSupervisadoIframe = document.getElementById('analisisNoSupervisadoIframe');
    
    // Referencias a los contenedores de spinner
    const visualizacionLoader = document.getElementById('visualizacionLoader');
    const analisisNoSupervisadoLoader = document.getElementById('analisisNoSupervisadoLoader');
    
    // Inicializar correctamente las alturas de los contenedores
    $('.module-container').css('height', '900px').css('padding-bottom', '0');
    
    // Función para manejar la carga de iframe
    function setupIframe(iframe, loader) {
        // Asegurarse de que el iframe sea visible y tenga el tamaño correcto
        iframe.style.height = '100%';
        iframe.style.width = '100%';
        iframe.style.display = 'block';
        
        // Configurar evento de carga
        iframe.onload = function() {
            // Ocultar el spinner de carga cuando el iframe está cargado
            loader.style.display = 'none';
            console.log(`Iframe ${iframe.id} cargado correctamente`);
            
            // Verificar si realmente se ha cargado contenido
            try {
                const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                const bodyElement = iframeDoc.body;
                
                if (!bodyElement || bodyElement.innerHTML.trim() === '') {
                    console.error(`Iframe ${iframe.id} tiene el body vacío`);
                    loader.innerHTML = '<div class="text-danger p-3">Error: No se pudo cargar el contenido</div>';
                    
                    // Intentar recargar automáticamente después de un tiempo
                    setTimeout(() => {
                        if (iframe.id === 'visualizacionIframe') {
                            iframe.src = '/visualizacion-dataset/iframe';
                        } else {
                            iframe.src = '/analisis-no-supervisado/iframe';
                        }
                        loader.innerHTML = '<div class="loader"></div>';
                        loader.style.display = 'flex';
                    }, 3000);
                }
            } catch (e) {
                console.error(`Error verificando el contenido del iframe ${iframe.id}:`, e);
            }
        };
        
        // Manejar errores
        iframe.onerror = function() {
            console.error(`Error al cargar el iframe ${iframe.id}`);
            // Mostrar mensaje de error en lugar del spinner
            loader.innerHTML = '<div class="text-danger p-3">Error al cargar el contenido</div>';
        };
    }
    
    // Configurar los iframes
    setupIframe(visualizacionIframe, visualizacionLoader);
    setupIframe(analisisNoSupervisadoIframe, analisisNoSupervisadoLoader);
    
    // Manejar cambios de pestaña
    $('a[data-bs-toggle="tab"], button[data-bs-toggle="tab"]').on('shown.bs.tab', function (e) {
        const targetId = $(e.target).attr('data-bs-target');
        
        if (targetId === '#visualizacion') {
            // Si la pestaña de visualización está vacía, recargar el iframe
            try {
                const iframeDoc = visualizacionIframe.contentDocument || visualizacionIframe.contentWindow.document;
                const bodyContent = iframeDoc.body.innerHTML.trim();
                
                if (bodyContent === '') {
                    visualizacionIframe.src = '/visualizacion-dataset/iframe';
                    visualizacionLoader.style.display = 'flex';
                }
            } catch (e) {
                // Si hay un error de acceso, es probable que el iframe esté vacío o con problemas
                visualizacionIframe.src = '/visualizacion-dataset/iframe';
                visualizacionLoader.style.display = 'flex';
            }
        } else if (targetId === '#analisis-no-supervisado') {
            // Si la pestaña de análisis no supervisado está vacía, recargar el iframe
            try {
                const iframeDoc = analisisNoSupervisadoIframe.contentDocument || analisisNoSupervisadoIframe.contentWindow.document;
                const bodyContent = iframeDoc.body.innerHTML.trim();
                
                if (bodyContent === '') {
                    analisisNoSupervisadoIframe.src = '/analisis-no-supervisado/iframe';
                    analisisNoSupervisadoLoader.style.display = 'flex';
                }
            } catch (e) {
                // Si hay un error de acceso, es probable que el iframe esté vacío o con problemas
                analisisNoSupervisadoIframe.src = '/analisis-no-supervisado/iframe';
                analisisNoSupervisadoLoader.style.display = 'flex';
            }
        }
    });
    
    // Garantizar que los iframes se carguen correctamente a pesar de posibles problemas
    setTimeout(function() {
        // Comprobar que el iframe de visualización se haya cargado correctamente
        try {
            const visDoc = visualizacionIframe.contentDocument || visualizacionIframe.contentWindow.document;
            if (!visDoc || !visDoc.body || visDoc.body.innerHTML.trim() === '') {
                console.log("Recargando iframe de visualización debido a carga incompleta");
                visualizacionIframe.src = '/visualizacion-dataset/iframe';
                visualizacionLoader.style.display = 'flex';
            } else {
                visualizacionLoader.style.display = 'none';
            }
        } catch (e) {
            console.error("Error verificando iframe de visualización:", e);
        }
        
        // Si después de un tiempo los spinners siguen visibles, ocultarlos
        setTimeout(function() {
            visualizacionLoader.style.display = 'none';
            analisisNoSupervisadoLoader.style.display = 'none';
        }, 5000);
    }, 3000);
});