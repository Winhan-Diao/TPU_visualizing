const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 25565;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));

// Path to the specific Python virtual environment
const PYTHON_PATH = '/media/winhan/Data/Develop/py/finance/.venv/bin/python';

// API endpoint to run TPU simulation and get data
app.post('/api/run-simulation', (req, res) => {
    const { n_layers = 24, n_nodes = 3, batch_size = 32, n_length = 3 } = req.body;
    
    // Execute Python wrapper script with parameters using the specific Python executable
    const python = spawn(PYTHON_PATH, [
        path.join(__dirname, 'my_tpu_ana_wrapper.py'),
        '--n_layers', n_layers.toString(),
        '--n_nodes', n_nodes.toString(),
        '--batch_size', batch_size.toString(),
        '--n_length', n_length.toString()
    ]);

    let data = '';
    
    python.stdout.on('data', (chunk) => {
        data += chunk.toString();
    });

    python.stderr.on('data', (err) => {
        console.error('Python error:', err.toString());
    });

    python.on('close', (code) => {
        if (code === 0) {
            try {
                const result = JSON.parse(data.trim());
                res.json(result);
            } catch (parseError) {
                console.error('JSON parse error:', parseError);
                res.status(500).json({ error: 'Failed to parse simulation results' });
            }
        } else {
            res.status(500).json({ error: 'Python script exited with error code: ' + code });
        }
    });
});

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});