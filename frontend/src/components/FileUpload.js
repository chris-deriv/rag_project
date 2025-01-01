import React, { useState } from 'react';
import {
  Button,
  Box,
  Typography,
  CircularProgress,
  Alert,
  Paper
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import api from '../api';

const FileUpload = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [currentFile, setCurrentFile] = useState(null);
  const [pollingCount, setPollingCount] = useState(0);
  const [processingStatus, setProcessingStatus] = useState(null);
  const MAX_POLLING_COUNT = 300; // 10 minutes max (300 * 2 seconds)

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setSuccess(false);
    setCurrentFile(null);
    setPollingCount(0);
    setProcessingStatus(null);

    try {
      // Upload the file
      const response = await api.uploadDocument(file);
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
      setSuccess(false);
      return;
    }

    try {
      const status = await api.getUploadStatus(filename);
      setProcessingStatus(status);
      
      switch (status.status) {
        case 'completed':
          setUploading(false);
          setSuccess(true);
          setError(null);
          if (onUploadSuccess) {
            onUploadSuccess();
          }
          break;
          
        case 'error':
          setError(status.error || 'Error processing document');
          setUploading(false);
          setSuccess(false);
          break;
          
        case 'processing':
          // Continue polling
          setPollingCount(prev => prev + 1);
          setTimeout(() => pollUploadStatus(filename), 2000);
          break;
          
        default:
          setError('Unknown processing status');
          setUploading(false);
          setSuccess(false);
      }
    } catch (err) {
      if (err.response?.status === 404) {
        // Document not found in processing list, check if it's in the document list
        try {
          const docs = await api.getDocumentNames();
          const doc = docs.find(d => d.source_name === filename);
          if (doc && doc.chunk_count > 0) {
            setUploading(false);
            setSuccess(true);
            setError(null);
            if (onUploadSuccess) {
              onUploadSuccess();
            }
            return;
          }
        } catch (listErr) {
          console.error('Error checking document list:', listErr);
        }
      }
      
      setError('Error checking upload status');
      setUploading(false);
      setSuccess(false);
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
        <Typography variant="h6" component="h2" sx={{ alignSelf: 'flex-start' }}>
          Upload Document
        </Typography>

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
            startIcon={<CloudUploadIcon />}
            disabled={uploading}
          >
            Choose File
          </Button>
        </label>

        {error && (
          <Alert severity="error" sx={{ width: '100%' }}>
            {error}
          </Alert>
        )}

        {success && !uploading && !error && (
          <Alert severity="success" sx={{ width: '100%' }}>
            Document processed successfully!
          </Alert>
        )}
      </Box>
    </Paper>
  );
};

export default FileUpload;
