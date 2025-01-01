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

const FileUpload = ({ onUploadSuccess, onUploadStart, onUploadError, isProcessing }) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [pollingCount, setPollingCount] = useState(0);
  const [processingStatus, setProcessingStatus] = useState(null);
  const MAX_POLLING_COUNT = 300; // 10 minutes max (300 * 2 seconds)

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setSuccess(false);
    setPollingCount(0);
    setProcessingStatus(null);

    try {
      // Upload the file
      const response = await api.uploadDocument(file);
      const filename = response.filename;
      
      // Notify parent that upload has started
      if (onUploadStart) {
        onUploadStart();
      }
      
      // Start polling for status
      pollUploadStatus(filename);
      
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Error uploading file';
      setError(errorMessage);
      setUploading(false);
      // Don't call onUploadError here since the file wasn't created yet
    }
    // Reset file input
    event.target.value = '';
  };

  const pollUploadStatus = async (filename) => {
    if (pollingCount >= MAX_POLLING_COUNT) {
      const timeoutError = 'Document processing timed out';
      setError(timeoutError);
      setUploading(false);
      setSuccess(false);
      if (onUploadError) {
        onUploadError(filename);
      }
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
          const errorMessage = status.error || 'Error processing document';
          setError(errorMessage);
          setUploading(false);
          setSuccess(false);
          if (onUploadError) {
            onUploadError(filename);
          }
          break;
          
        case 'processing':
          // Continue polling
          setPollingCount(prev => prev + 1);
          setTimeout(() => pollUploadStatus(filename), 2000);
          break;
          
        default:
          const unknownError = 'Unknown processing status';
          setError(unknownError);
          setUploading(false);
          setSuccess(false);
          if (onUploadError) {
            onUploadError(filename);
          }
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
      
      const statusError = 'Error checking upload status';
      setError(statusError);
      setUploading(false);
      setSuccess(false);
      if (onUploadError) {
        onUploadError(filename);
      }
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
          disabled={uploading || isProcessing}
        />

        <label htmlFor="file-upload">
          <Button
            variant="contained"
            component="span"
            startIcon={uploading || isProcessing ? <CircularProgress size={20} /> : <CloudUploadIcon />}
            disabled={uploading || isProcessing}
          >
            {uploading ? 'Uploading...' : isProcessing ? 'Processing Document...' : 'Choose File'}
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

        {processingStatus && processingStatus.status === 'processing' && (
          <Alert severity="info" sx={{ width: '100%' }}>
            Processing document... {processingStatus.progress || ''}
          </Alert>
        )}
      </Box>
    </Paper>
  );
};

export default FileUpload;
