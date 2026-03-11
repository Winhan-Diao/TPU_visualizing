const express = require('express');
const { spawn } = require('child_process');
const { animate } = require('animejs');
const path = require('path');
const cors = require('cors');
const fs = require('fs'); // Added for file system operations

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

// Endpoint to delete the details file after it's been read
app.delete('/api/delete/:filename', (req, res) => {
    const filename = req.params.filename;
    const filePath = path.join(__dirname, 'public', filename);
    
    // Security check to prevent directory traversal
    const publicDir = path.resolve(__dirname, 'public');
    const resolvedPath = path.resolve(filePath);
    
    if (!resolvedPath.startsWith(publicDir)) {
        return res.status(400).json({ error: 'Invalid file path' });
    }
    
    fs.unlink(filePath, (err) => {
        if (err) {
            if (err.code === 'ENOENT') {
                // File doesn't exist, which is fine
                console.log(`File ${filePath} was already deleted or never existed`);
                return res.status(200).json({ message: 'File already deleted or does not exist' });
            } else {
                console.error('Error deleting file:', err);
                return res.status(500).json({ error: 'Could not delete file' });
            }
        }
        console.log(`Successfully deleted file: ${filePath}`);
        res.status(200).json({ message: 'File deleted successfully' });
    });
});

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});