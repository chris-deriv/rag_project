import React, { useState } from 'react';
import {
  Button,
  Box,
  Typography,
  CircularProgress,
  Alert,
  Paper,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DeleteIcon from '@mui/icons-material/Delete';
import api from '../api';

const FileUpload = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [currentFile, setCurrentFile] = useState(null);
  const [pollingCount, setPollingCount] = useState(0);
  const MAX_POLLING_COUNT = 60; // 2 minutes max (60 * 2 seconds)

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setSuccess(false);
    setCurrentFile(null);
    setPollingCount(0);

    try {
      // Upload the file
      const response = await api.uploadDocument(file);
      setSuccess(true);
      setCurrentFile(response.filename);
      
      // Start polling for status
      pollUploadStatus(response.filename);
      
    } catch (err) {
      setError(err.response?.data?.error || 'Error uploading file');
      setUploading(false);
    }
    // Reset file input
    event.target.value = '';
  };

  const pollUploadStatus = async (filename) => {
    if (pollingCount >= MAX_POLLING_COUNT) {
      setError('Document processing timed out');
      setUploading(false);
      return;
    }

    try {
      const status = await api.getUploadStatus(filename);
      
      if (status.status === 'completed') {
        setUploading(false);
        // Get final document list
        const docs = await api.getDocumentNames();
        if (onUploadSuccess) {
          onUploadSuccess(docs);
        }
      } else if (status.status === 'error') {
        setError(status.error || 'Error processing document');
        setUploading(false);
      } else if (status.status === 'processing') {
        // Continue polling
        setPollingCount(prev => prev + 1);
        setTimeout(() => pollUploadStatus(filename), 2000);
      }
    } catch (err) {
      if (err.response?.status === 404) {
        // Document not found in processing list, try getting document list
        try {
          const docs = await api.getDocumentNames();
          const doc = docs.find(d => d.source_name === filename);
          if (doc) {
            setUploading(false);
            if (onUploadSuccess) {
              onUploadSuccess(docs);
            }
            return;
          }
        } catch (listErr) {
          console.error('Error checking document list:', listErr);
        }
      }
      
      setError('Error checking upload status');
      setUploading(false);
    }
  };

  const handleReset = async () => {
    setConfirmOpen(false);
    setResetting(true);
    setError(null);
    try {
      await api.resetDatabase();
      setSuccess(true);
      // Get updated document list after reset
      const docs = await api.getDocumentNames();
      if (onUploadSuccess) {
        onUploadSuccess(docs);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Error resetting database');
    } finally {
      setResetting(false);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3, mb: 3 }}>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 2
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
          <Typography variant="h6" component="h2">
            Upload Document
          </Typography>
          <Button
            variant="contained"
            color="error"
            startIcon={resetting ? <CircularProgress size={20} /> : <DeleteIcon />}
            onClick={() => setConfirmOpen(true)}
            disabled={uploading || resetting}
          >
            Reset Database
          </Button>
        </Box>

        <input
          accept=".pdf,.doc,.docx"
          style={{ display: 'none' }}
          id="file-upload"
          type="file"
          onChange={handleFileChange}
          disabled={uploading}
        />

        <label htmlFor="file-upload">
          <Button
            variant="contained"
            component="span"
            startIcon={uploading ? <CircularProgress size={20} /> : <CloudUploadIcon />}
            disabled={uploading}
          >
            {uploading ? 'Processing Document...' : 'Choose File'}
          </Button>
        </label>

        {error && (
          <Alert severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        )}

        {success && !uploading && (
          <Alert severity="success" sx={{ width: '100%' }}>
            File uploaded successfully!
          </Alert>
        )}

        {uploading && (
          <Alert severity="info" sx={{ width: '100%' }}>
            Processing document... This may take a moment.
            {currentFile && (
              <Typography variant="caption" display="block">
                File: {currentFile}
                {pollingCount > 0 && ` (${Math.round((pollingCount / MAX_POLLING_COUNT) * 100)}%)`}
              </Typography>
            )}
          </Alert>
        )}
      </Box>

      {/* Confirmation Dialog */}
      <Dialog open={confirmOpen} onClose={() => setConfirmOpen(false)}>
        <DialogTitle>Confirm Reset</DialogTitle>
        <DialogContent>
          Are you sure you want to reset the database? This will delete all uploaded documents and cannot be undone.
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmOpen(false)}>Cancel</Button>
          <Button
            onClick={handleReset}
            color="error"
            variant="contained"
          >
            Reset
          </Button>
        </DialogActions>
      </Dialog>
    </Paper>
  );
};

export default FileUpload;
