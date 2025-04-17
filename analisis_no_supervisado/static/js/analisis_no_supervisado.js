$(document).ready(function () {
    const uploadForm = document.getElementById('uploadForm');
    const loadingContainer = document.getElementById('loadingContainer');
    const resultContainer = document.getElementById('resultContainer');
    const progressBar = document.getElementById('progressBar');
    const estimatedTimeEl = document.getElementById('estimatedTime');
    const resetBtn = document.getElementById('btnReset');
    const fileInput = document.getElementById('file');
    const columnsToOmitSelect = document.getElementById('columnsToOmit');
    let dataTable;

    // Inicializar Select2 desde el principio con configuración básica
    // para evitar que se muestre abierto inicialmente
    $(columnsToOmitSelect).select2({
        placeholder: 'Seleccione columnas a omitir',
        allowClear: true,
        width: '100%',
        theme: 'bootstrap-5',
        dropdownCssClass: 'select2-dropdown-custom', // Clase personalizada para el dropdown
        selectionCssClass: 'select2-selection-custom' // Clase personalizada para la selección
    });

    // Resto del código...

    // Función para resetear el formulario y la visualización
    function resetForm() {
        // Resetear el formulario
        uploadForm.reset();

        // Ocultar los contenedores de resultados y carga
        loadingContainer.classList.add('d-none');
        resultContainer.classList.add('d-none');

        // Si hay una tabla de datos inicializada, destruirla
        if (dataTable) {
            dataTable.destroy();
            dataTable = null;
        }

        // Limpiar los encabezados y cuerpo de la tabla
        document.getElementById('dataTableHead').innerHTML = '';
        document.getElementById('dataTableBody').innerHTML = '';

        // Resetear select2 en lugar de solo limpiar HTML
        $(columnsToOmitSelect).empty();
        $(columnsToOmitSelect).select2('destroy');
        $(columnsToOmitSelect).select2({
            placeholder: 'Seleccione columnas a omitir',
            allowClear: true,
            width: '100%',
            theme: 'bootstrap-5',
            dropdownCssClass: 'select2-dropdown-custom',
            selectionCssClass: 'select2-selection-custom'
        });

        // Scroll hacia arriba
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Event listener para el botón de reset
    resetBtn.addEventListener('click', resetForm);

    // Event listener para cuando se selecciona un archivo
    fileInput.addEventListener('change', function (e) {
        // No inicializar Select2 aquí, ya se hizo al cargar la página
        // Solo se actualizará con nuevos datos

        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];

            // Mostrar spinner de carga pequeño sólo para la carga del archivo
            loadingContainer.classList.remove('d-none');
            resultContainer.classList.add('d-none');
            progressBar.style.width = '50%';
            progressBar.setAttribute('aria-valuenow', 50);
            estimatedTimeEl.textContent = 'Cargando vista previa...';

            // Crear FormData solo con el archivo
            const formData = new FormData();
            formData.append('file', file);
            formData.append('preview_only', 'true');  // Indicar que solo queremos vista previa

            // Enviar solicitud para vista previa
            fetch('/analisis-no-supervisado/preview/', {
                method: 'POST',
                body: formData
            })
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(data => {
                            throw new Error(data.error || 'Error al procesar el archivo');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    // Ocultar spinner
                    loadingContainer.classList.add('d-none');
                    resultContainer.classList.remove('d-none');

                    // Mostrar solo la pestaña de datos
                    $('#analysisTabs a[href="#data"]').tab('show');

                    // Ocultar otras pestañas hasta que se haga el análisis completo
                    document.querySelectorAll('#analysisTabs button:not(#data-tab)').forEach(tab => {
                        tab.setAttribute('disabled', 'disabled');
                        tab.classList.add('disabled');
                    });

                    // Llenar información básica mínima
                    document.getElementById('fileName').textContent = data.fileName;
                    document.getElementById('rowCount').textContent = data.rowCount;
                    document.getElementById('columnCount').textContent = data.columnCount;
                    document.getElementById('numericColumnCount').textContent = 'No analizado aún';

                    // Resetear y actualizar las columnas en el select múltiple
                    $(columnsToOmitSelect).empty();  // Vaciar primero

                    // Añadir nuevas opciones
                    data.columns.forEach(column => {
                        const option = document.createElement('option');
                        option.value = column;
                        option.textContent = column;
                        columnsToOmitSelect.appendChild(option);
                    });

                    // Refrescar Select2 para que muestre las nuevas opciones correctamente
                    $(columnsToOmitSelect).trigger('change');

                    // Llenar tabla de datos
                    const tableHead = document.getElementById('dataTableHead');
                    const tableBody = document.getElementById('dataTableBody');

                    // Crear encabezados
                    let headerRow = '<tr>';
                    data.columns.forEach(column => {
                        headerRow += `<th>${column}<br><small class="text-muted">${data.dtypes[column]}</small></th>`;
                    });
                    headerRow += '</tr>';
                    tableHead.innerHTML = headerRow;

                    // Llenar cuerpo de la tabla
                    tableBody.innerHTML = '';
                    data.tableData.forEach(row => {
                        let rowHtml = '<tr>';
                        data.columns.forEach(column => {
                            rowHtml += `<td>${row[column] !== null ? row[column] : '<span class="text-muted">null</span>'}</td>`;
                        });
                        rowHtml += '</tr>';
                        tableBody.innerHTML += rowHtml;
                    });

                    // Inicializar DataTable
                    if (dataTable) {
                        dataTable.destroy();
                    }
                    dataTable = $('#dataTable').DataTable({
                        pageLength: 10,
                        scrollX: true,
                        responsive: true
                    });

                    // Scroll a los resultados
                    resultContainer.scrollIntoView({ behavior: 'smooth' });
                })
                .catch(error => {
                    // Manejar errores
                    loadingContainer.classList.add('d-none');
                    alert(`Error al cargar la vista previa: ${error.message}`);
                    console.error('Error:', error);
                });
        }
    });

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();

        // Mostrar spinner de carga
        loadingContainer.classList.remove('d-none');

        // Obtener el formulario y preparar para envío
        const formData = new FormData();

        // Añadir el archivo
        formData.append('file', fileInput.files[0]);

        // Añadir el límite de filas si existe
        const maxRows = document.getElementById('maxRows').value;
        if (maxRows) {
            formData.append('maxRows', maxRows);
        }

        // Añadir las columnas a omitir (pueden ser múltiples)
        const selectedColumns = $(columnsToOmitSelect).val();
        if (selectedColumns && selectedColumns.length > 0) {
            // Para enviar múltiples valores con el mismo nombre
            selectedColumns.forEach(column => {
                formData.append('columnsToOmit', column);
            });
        }

        const file = fileInput.files[0];

        // Estimar tiempo basado en tamaño del archivo
        let estimatedTime = Math.max(3, file.size / (1024 * 1024) * 0.5); // ~0.5 segundo por MB, mínimo 3 segundos
        estimatedTimeEl.textContent = `Tiempo estimado: ${estimatedTime.toFixed(1)} segundos`;

        // Simular progreso
        let progress = 0;
        const interval = setInterval(() => {
            progress += 100 / (estimatedTime * 10); // actualizar cada 100ms
            if (progress >= 95) {
                progress = 95; // dejar el 5% para el final
                clearInterval(interval);
            }
            progressBar.style.width = `${progress}%`;
            progressBar.setAttribute('aria-valuenow', progress);
        }, 100);

        // Enviar solicitud
        fetch('/analisis-no-supervisado/', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                // Completar la barra de progreso
                clearInterval(interval);
                progressBar.style.width = '100%';
                progressBar.setAttribute('aria-valuenow', 100);

                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Error al procesar el archivo');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Ocultar spinner y mostrar resultados
                loadingContainer.classList.add('d-none');
                resultContainer.classList.remove('d-none');

                // Habilitar todas las pestañas
                document.querySelectorAll('#analysisTabs button').forEach(tab => {
                    tab.removeAttribute('disabled');
                    tab.classList.remove('disabled');
                });

                // Llenar información básica
                document.getElementById('fileName').textContent = data.fileName;
                document.getElementById('rowCount').textContent = data.rowCount;
                if (data.rowsLimited) {
                    document.getElementById('rowCount').textContent += ` (limitado a las primeras ${data.maxRowsApplied} filas)`;
                }
                document.getElementById('columnCount').textContent = data.columnCount;
                document.getElementById('numericColumnCount').textContent = data.numericColumnCount;

                // Información y descripción
                document.getElementById('dataInfo').textContent = data.info;
                document.getElementById('dataDescribe').innerHTML = data.describe;

                // Llenar tabla de datos (mantener la que ya existía si no hay cambios)
                if (dataTable) {
                    dataTable.destroy();
                }

                const tableHead = document.getElementById('dataTableHead');
                const tableBody = document.getElementById('dataTableBody');

                // Crear encabezados
                let headerRow = '<tr>';
                data.columns.forEach(column => {
                    headerRow += `<th>${column}<br><small class="text-muted">${data.dtypes[column]}</small></th>`;
                });
                headerRow += '</tr>';
                tableHead.innerHTML = headerRow;

                // Llenar cuerpo de la tabla
                tableBody.innerHTML = '';
                data.tableData.forEach(row => {
                    let rowHtml = '<tr>';
                    data.columns.forEach(column => {
                        rowHtml += `<td>${row[column] !== null ? row[column] : '<span class="text-muted">null</span>'}</td>`;
                    });
                    rowHtml += '</tr>';
                    tableBody.innerHTML += rowHtml;
                });

                // Inicializar DataTable
                dataTable = $('#dataTable').DataTable({
                    pageLength: 10,
                    scrollX: true,
                    responsive: true
                });

                // Mostrar gráficos
                document.getElementById('correlationPlot').src = 'data:image/png;base64,' + data.plotCorrelation;

                // Llenar tabla de valores faltantes
                const missingValuesBody = document.getElementById('missingValuesBody');
                missingValuesBody.innerHTML = '';
                Object.keys(data.missingStats).forEach(column => {
                    const count = data.missingStats[column].count;
                    const percent = data.missingStats[column].percent.toFixed(2);
                    missingValuesBody.innerHTML += `
                    <tr>
                        <td>${column}</td>
                        <td>${count}</td>
                        <td>${percent}%</td>
                        <td>
                            <div class="progress">
                                <div class="progress-bar bg-warning" role="progressbar" style="width: ${percent}%" 
                                    aria-valuenow="${percent}" aria-valuemin="0" aria-valuemax="100"></div>
                            </div>
                        </td>
                    </tr>`;
                });

                // Mostrar histogramas
                const histograms = document.getElementById('histograms');
                histograms.innerHTML = '';
                data.histograms.forEach(hist => {
                    histograms.innerHTML += `
                    <div class="col-md-6 mb-4">
                        <h6>${hist.title}</h6>
                        <img src="data:image/png;base64,${hist.image}" class="img-fluid">
                    </div>`;
                });

                // Mostrar gráficos de barras para variables categóricas
                const barplots = document.getElementById('barplots');
                barplots.innerHTML = '';
                data.barplots.forEach(plot => {
                    barplots.innerHTML += `
                    <div class="col-md-6 mb-4">
                        <h6>${plot.title}</h6>
                        <img src="data:image/png;base64,${plot.image}" class="img-fluid">
                    </div>`;
                });

                // Mostrar correlaciones más fuertes
                const topCorrelationsBody = document.getElementById('topCorrelationsBody');
                topCorrelationsBody.innerHTML = '';
                data.topCorrelations.forEach(corr => {
                    const corrValue = corr.value.toFixed(3);
                    const absValue = Math.abs(corr.value);
                    const color = corr.value > 0 ? 'success' : 'danger';

                    topCorrelationsBody.innerHTML += `
                    <tr>
                        <td>${corr.var1}</td>
                        <td>${corr.var2}</td>
                        <td class="text-${color}">${corrValue}</td>
                        <td>
                            <div class="progress">
                                <div class="progress-bar bg-${color}" role="progressbar" 
                                    style="width: ${absValue * 100}%" aria-valuenow="${absValue * 100}" 
                                    aria-valuemin="0" aria-valuemax="100"></div>
                            </div>
                        </td>
                    </tr>`;
                });

                // Mostrar matriz de dispersión
                const scatterPlots = document.getElementById('scatterPlots');
                if (data.scatterMatrix) {
                    scatterPlots.innerHTML = `
                    <div class="mb-4">
                        <h6>Matriz de Dispersión (Variables Principales)</h6>
                        <img src="data:image/png;base64,${data.scatterMatrix}" class="img-fluid" style="max-width: 800px;">
                    </div>`;
                } else {
                    scatterPlots.innerHTML = '<p>No hay suficientes variables numéricas para generar la matriz de dispersión.</p>';
                }

                // Mostrar outliers
                const outliersTableBody = document.getElementById('outliersTableBody');
                outliersTableBody.innerHTML = '';
                data.outliers.forEach(outlier => {
                    const percent = outlier.percentage.toFixed(2);
                    outliersTableBody.innerHTML += `
                    <tr>
                        <td>${outlier.feature}</td>
                        <td>${outlier.count}</td>
                        <td>${percent}%</td>
                        <td>
                            <div class="progress">
                                <div class="progress-bar bg-danger" role="progressbar" 
                                    style="width: ${percent}%" aria-valuenow="${percent}" 
                                    aria-valuemin="0" aria-valuemax="100"></div>
                            </div>
                        </td>
                    </tr>`;
                });

                // Mostrar boxplots
                const boxplots = document.getElementById('boxplots');
                boxplots.innerHTML = '';
                data.boxplots.forEach(boxplot => {
                    boxplots.innerHTML += `
                    <div class="col-md-6 mb-4">
                        <h6>${boxplot.title}</h6>
                        <img src="data:image/png;base64,${boxplot.image}" class="img-fluid">
                    </div>`;
                });

                // Llenar tabla de sugerencias de preprocesamiento
                const preprocessingTableBody = document.getElementById('preprocessingTableBody');
                preprocessingTableBody.innerHTML = '';
                Object.keys(data.preprocessingSuggestions).forEach(column => {
                    const suggestions = data.preprocessingSuggestions[column];
                    const suggestionsHtml = suggestions.map(s => `<li>${s}</li>`).join('');

                    preprocessingTableBody.innerHTML += `
                    <tr>
                        <td>${column}</td>
                        <td>${data.dtypes[column]}</td>
                        <td>
                            <ul>
                                ${suggestionsHtml || '<li>No hay sugerencias específicas</li>'}
                            </ul>
                        </td>
                    </tr>`;
                });

                // Mostrar información de dimensionalidad
                document.getElementById('dimensionalityValue').textContent = data.dimensionalityInfo;

                // Hacer scroll a los resultados
                setTimeout(() => {
                    resultContainer.scrollIntoView({ behavior: 'smooth' });
                }, 100);

                // Configurar el botón para nuevo análisis
                document.getElementById('btnNuevoAnalisis').addEventListener('click', resetForm);
            })
            .catch(error => {
                // Manejar errores
                loadingContainer.classList.add('d-none');
                resultContainer.classList.add('d-none');

                console.error('Error completo:', error);

                if (error.message) {
                    alert(`Error: ${error.message}`);
                } else {
                    alert('Error desconocido al procesar el archivo');
                }
                console.error('Error:', error);
            });
    });
});