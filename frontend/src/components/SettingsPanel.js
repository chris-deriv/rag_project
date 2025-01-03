import React, { useState, useEffect } from 'react';
import api from '../api';
import {
    TextField,
    Select,
    MenuItem,
    FormControl,
    InputLabel,
    Typography,
    Button,
    Box,
    Alert,
    Paper
} from '@mui/material';

const SettingsPanel = ({ onClose }) => {
    const [settings, setSettings] = useState(null);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadSettings();
    }, []);

    const loadSettings = async () => {
        try {
            setLoading(true);
            const data = await api.getSettings();
            setSettings(data);
            setError(null);
        } catch (err) {
            setError('Failed to load settings');
            console.error('Error loading settings:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await api.updateSettings(settings);
            if (response.error) {
                setError(response.error);
                setSuccess(null);
            } else {
                setSuccess('Settings updated successfully');
                setError(null);
                // Reset success message after 3 seconds
                setTimeout(() => setSuccess(null), 3000);
                if (onClose) {
                    setTimeout(onClose, 1000);
                }
            }
        } catch (err) {
            setError('Failed to update settings');
            setSuccess(null);
            console.error('Error updating settings:', err);
        }
    };

    const handleChange = (category, field, value) => {
        setSettings(prev => ({
            ...prev,
            [category]: {
                ...prev[category],
                [field]: value
            }
        }));
    };

    if (loading) return <Typography>Loading settings...</Typography>;
    if (!settings) return <Typography>No settings available</Typography>;

    return (
        <Box component="form" onSubmit={handleSubmit} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>Settings</Typography>
            
            {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
            {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}
            
            {/* LLM Settings */}
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle1" gutterBottom>LLM Settings</Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                        label="Temperature"
                        type="number"
                        inputProps={{ step: 0.1, min: 0, max: 2 }}
                        value={settings.llm.temperature}
                        onChange={(e) => handleChange('llm', 'temperature', parseFloat(e.target.value))}
                        helperText="Controls randomness in responses (0-2)"
                        fullWidth
                    />
                    <TextField
                        label="Max Tokens"
                        type="number"
                        inputProps={{ min: 1 }}
                        value={settings.llm.max_tokens}
                        onChange={(e) => handleChange('llm', 'max_tokens', parseInt(e.target.value))}
                        helperText="Maximum length of generated responses"
                        fullWidth
                    />
                    <FormControl fullWidth>
                        <InputLabel>Model</InputLabel>
                        <Select
                            value={settings.llm.model}
                            onChange={(e) => handleChange('llm', 'model', e.target.value)}
                            label="Model"
                        >
                            <MenuItem value="gpt-3.5-turbo">GPT-3.5 Turbo</MenuItem>
                            <MenuItem value="gpt-4">GPT-4</MenuItem>
                        </Select>
                    </FormControl>
                </Box>
            </Paper>

            {/* Document Processing Settings */}
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle1" gutterBottom>Document Processing</Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                        label="Chunk Size"
                        type="number"
                        inputProps={{ min: 100 }}
                        value={settings.document_processing.chunk_size}
                        onChange={(e) => handleChange('document_processing', 'chunk_size', parseInt(e.target.value))}
                        helperText="Size of text chunks for processing"
                        fullWidth
                    />
                    <TextField
                        label="Chunk Overlap"
                        type="number"
                        inputProps={{ min: 0 }}
                        value={settings.document_processing.chunk_overlap}
                        onChange={(e) => handleChange('document_processing', 'chunk_overlap', parseInt(e.target.value))}
                        helperText="Overlap between consecutive chunks"
                        fullWidth
                    />
                </Box>
            </Paper>

            {/* Response Settings */}
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography variant="subtitle1" gutterBottom>Response Settings</Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                        label="System Prompt"
                        multiline
                        rows={4}
                        value={settings.response.system_prompt}
                        onChange={(e) => handleChange('response', 'system_prompt', e.target.value)}
                        helperText="Base prompt for response generation"
                        fullWidth
                    />
                    <TextField
                        label="Source Citation Prompt"
                        multiline
                        rows={4}
                        value={settings.response.source_citation_prompt}
                        onChange={(e) => handleChange('response', 'source_citation_prompt', e.target.value)}
                        helperText="Prompt for responses with source citations"
                        fullWidth
                    />
                </Box>
            </Paper>

            <Box sx={{ display: 'flex', gap: 2, justifyContent: 'flex-end' }}>
                <Button variant="outlined" onClick={loadSettings}>
                    Reset Changes
                </Button>
                <Button variant="contained" type="submit" color="primary">
                    Save Settings
                </Button>
            </Box>
        </Box>
    );
};

export default SettingsPanel;
