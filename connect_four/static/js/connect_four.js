document.addEventListener('DOMContentLoaded', function() {
    // Constantes
    const ROWS = 6;
    const COLS = 7;
    const EMPTY = '.';
    const PLAYER = 'O';
    const AI = 'X';
    
    // Variables del juego
    let board = [];
    let currentPlayer = PLAYER;
    let gameOver = false;
    let gameStartTime = new Date();
    let gameTimer;
    let difficulty = 'normal';
    
    // Referencias a elementos del DOM
    const gameBoard = document.getElementById('gameBoard');
    const turnIndicator = document.getElementById('turnIndicator');
    const moveCount = document.getElementById('moveCount');
    const gameTime = document.getElementById('gameTime');
    const winnerMessage = document.getElementById('winnerMessage');
    const resetButton = document.getElementById('resetButton');
    const aiFirstButton = document.getElementById('aiFirstButton');
    const columnSelectors = document.querySelectorAll('.column-selector');
    
    // Inicializar tablero
    function initBoard() {
        board = [];
        
        // Crear array 2D para el tablero
        for (let r = 0; r < ROWS; r++) {
            let row = [];
            for (let c = 0; c < COLS; c++) {
                row.push(EMPTY);
            }
            board.push(row);
        }
        
        // Limpiar el tablero visual
        gameBoard.innerHTML = '';
        
        // Crear las celdas visuales
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.row = r;
                cell.dataset.col = c;
                gameBoard.appendChild(cell);
            }
        }
        
        // Reiniciar variables del juego
        currentPlayer = PLAYER;
        gameOver = false;
        winnerMessage.style.display = 'none';
        turnIndicator.className = 'turn-indicator player-turn';
        turnIndicator.textContent = 'Tu turno';
        moveCount.textContent = '0';
        
        // Reiniciar temporizador
        clearInterval(gameTimer);
        gameStartTime = new Date();
        gameTimer = setInterval(updateGameTime, 1000);
        gameTime.textContent = '00:00';
        
        // Habilitar interacciones
        enableColumnSelectors();
    }
    
    // Actualizar el temporizador del juego
    function updateGameTime() {
        const now = new Date();
        const diff = Math.floor((now - gameStartTime) / 1000);
        const minutes = Math.floor(diff / 60).toString().padStart(2, '0');
        const seconds = (diff % 60).toString().padStart(2, '0');
        gameTime.textContent = `${minutes}:${seconds}`;
    }
    
    // Habilitar selectores de columna
    function enableColumnSelectors() {
        columnSelectors.forEach(selector => {
            selector.style.visibility = 'visible';
        });
    }
    
    // Deshabilitar selectores de columna
    function disableColumnSelectors() {
        columnSelectors.forEach(selector => {
            selector.style.visibility = 'hidden';
        });
    }
    
    // Realizar un movimiento
    function makeMove(col) {
        if (gameOver || currentPlayer !== PLAYER) return;
        
        // Encontrar la primera celda vacía en la columna (de abajo hacia arriba)
        for (let r = ROWS - 1; r >= 0; r--) {
            if (board[r][col] === EMPTY) {
                board[r][col] = currentPlayer;
                updateBoardUI();
                moveCount.textContent = parseInt(moveCount.textContent) + 1;
                
                // Verificar victoria
                if (checkWin(r, col, currentPlayer)) {
                    gameOver = true;
                    showWinner('¡Has ganado!');
                    disableColumnSelectors();
                    clearInterval(gameTimer);
                    return;
                }
                
                // Verificar empate
                if (checkDraw()) {
                    gameOver = true;
                    showWinner('¡Empate!');
                    disableColumnSelectors();
                    clearInterval(gameTimer);
                    return;
                }
                
                // Cambiar turno
                currentPlayer = AI;
                turnIndicator.className = 'turn-indicator ai-turn';
                turnIndicator.textContent = 'Turno de la IA...';
                
                // Hacer que la IA juegue después de un breve retraso
                setTimeout(aiMove, 700);
                return;
            }
        }
    }
    
    // Movimiento de la IA
    function aiMove() {
        if (gameOver) return;
        
        // Obtener estado actual del tablero para enviar al modelo
        const boardState = {};
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const index = r * COLS + c;
                boardState[index] = board[r][c];
            }
        }
        
        // Siempre usar el modelo ML para predecir el movimiento
        // En modo difícil, usamos parámetros más avanzados
        fetch('/connect-four/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                board_state: boardState,
                difficulty: difficulty
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                makeAIMove(data.move);
            } else {
                console.error('Error en la predicción:', data.error);
                // Si hay error, elegir una columna disponible aleatoria
                makeRandomMove();
            }
        })
        .catch(error => {
            console.error('Error al llamar a la API:', error);
            // Si hay error, elegir una columna disponible aleatoria
            makeRandomMove();
        });
    }
    
    // Movimiento aleatorio como fallback en caso de error
    function makeRandomMove() {
        const validColumns = [];
        
        // Identificar columnas disponibles
        for (let c = 0; c < COLS; c++) {
            if (board[0][c] === EMPTY) {
                validColumns.push(c);
            }
        }
        
        if (validColumns.length > 0) {
            // Elegir columna aleatoria entre las disponibles
            const randomColumn = validColumns[Math.floor(Math.random() * validColumns.length)];
            makeAIMove(randomColumn);
        } else {
            // Si no hay columnas disponibles, es un empate
            gameOver = true;
            showWinner('¡Empate!');
            disableColumnSelectors();
            clearInterval(gameTimer);
        }
    }
    
    // Ejecutar el movimiento elegido por la IA
    function makeAIMove(col) {
        // Encontrar la primera celda vacía en la columna (de abajo hacia arriba)
        for (let r = ROWS - 1; r >= 0; r--) {
            if (board[r][col] === EMPTY) {
                board[r][col] = AI;
                updateBoardUI();
                moveCount.textContent = parseInt(moveCount.textContent) + 1;
                
                // Verificar victoria
                if (checkWin(r, col, AI)) {
                    gameOver = true;
                    showWinner('La IA ha ganado');
                    disableColumnSelectors();
                    clearInterval(gameTimer);
                    return;
                }
                
                // Verificar empate
                if (checkDraw()) {
                    gameOver = true;
                    showWinner('¡Empate!');
                    disableColumnSelectors();
                    clearInterval(gameTimer);
                    return;
                }
                
                // Cambiar turno
                currentPlayer = PLAYER;
                turnIndicator.className = 'turn-indicator player-turn';
                turnIndicator.textContent = 'Tu turno';
                return;
            }
        }
    }
    
    // Actualizar la interfaz del tablero
    function updateBoardUI() {
        const cells = gameBoard.querySelectorAll('.cell');
        
        cells.forEach(cell => {
            const row = parseInt(cell.dataset.row);
            const col = parseInt(cell.dataset.col);
            
            // Eliminar clases previas
            cell.classList.remove('player', 'ai', 'animate-drop');
            void cell.offsetWidth; // Forzar reflow para reiniciar la animación
            
            // Agregar clase según el estado de la celda
            if (board[row][col] === PLAYER) {
                cell.classList.add('player', 'animate-drop');
            } else if (board[row][col] === AI) {
                cell.classList.add('ai', 'animate-drop');
            }
        });
    }
    
    // Verificar victoria
    function checkWin(row, col, player) {
        // Verificar horizontal
        let count = 0;
        for (let c = 0; c < COLS; c++) {
            if (board[row][c] === player) {
                count++;
                if (count >= 4) return true;
            } else {
                count = 0;
            }
        }
        
        // Verificar vertical
        count = 0;
        for (let r = 0; r < ROWS; r++) {
            if (board[r][col] === player) {
                count++;
                if (count >= 4) return true;
            } else {
                count = 0;
            }
        }
        
        // Verificar diagonal /
        count = 0;
        let startRow = row + Math.min(col, row);
        let startCol = col - Math.min(col, row);
        
        while (startRow >= 0 && startCol < COLS) {
            if (startRow < ROWS && board[startRow][startCol] === player) {
                count++;
                if (count >= 4) return true;
            } else {
                count = 0;
            }
            startRow--;
            startCol++;
        }
        
        // Verificar diagonal \
        count = 0;
        startRow = row - Math.min(col, ROWS - 1 - row);
        startCol = col - Math.min(col, ROWS - 1 - row);
        
        while (startRow < ROWS && startCol < COLS) {
            if (startRow >= 0 && board[startRow][startCol] === player) {
                count++;
                if (count >= 4) return true;
            } else {
                count = 0;
            }
            startRow++;
            startCol++;
        }
        
        return false;
    }
    
    // Verificar empate
    function checkDraw() {
        for (let c = 0; c < COLS; c++) {
            if (board[0][c] === EMPTY) {
                return false;
            }
        }
        return true;
    }
    
    // Mostrar mensaje de ganador
    function showWinner(message) {
        winnerMessage.textContent = message;
        winnerMessage.style.display = 'block';
    }
    
    // Eventos para los selectores de columna
    columnSelectors.forEach(selector => {
        selector.addEventListener('click', function() {
            if (!gameOver && currentPlayer === PLAYER) {
                const col = parseInt(this.dataset.column);
                makeMove(col);
            }
        });
    });
    
    // Evento para el botón de reinicio
    resetButton.addEventListener('click', function() {
        initBoard();
    });
    
    // Evento para el botón de IA primero
    aiFirstButton.addEventListener('click', function() {
        initBoard();
        currentPlayer = AI;
        turnIndicator.className = 'turn-indicator ai-turn';
        turnIndicator.textContent = 'Turno de la IA...';
        
        // Hacer que la IA juegue después de un breve retraso
        setTimeout(aiMove, 700);
    });
    
    // Eventos para los selectores de dificultad
    document.getElementById('difficultySelector').addEventListener('change', function() {
        // Actualizar dificultad
        difficulty = this.value;
        
        // Si hay un juego en curso, mostrar mensaje
        if (!gameOver && parseInt(moveCount.textContent) > 0) {
            alert('La dificultad se aplicará en el próximo juego.');
        }
    });
    
    // Inicializar el juego
    initBoard();
});