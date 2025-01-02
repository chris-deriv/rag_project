import React, { useState, useEffect } from 'react';
import api from './api';
import { Container, Grid, Paper, Box, Button, Dialog, DialogTitle, DialogContent, DialogActions, Typography } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import FileUpload from './components/FileUpload';
import DocumentSearch from './components/DocumentSearch';
import ChatInterface from './components/ChatInterface';
import SettingsPanel from './components/SettingsPanel';

function App() {
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [processingDocument, setProcessingDocument] = useState(false);

  // Track document processing status
  useEffect(() => {
    let pollInterval;
    const checkProcessingStatus = async () => {
      try {
        const docs = await api.getDocumentNames();
        const hasProcessing = docs.some(doc => doc.status === 'processing');
        setProcessingDocument(hasProcessing);
        
        if (!hasProcessing && pollInterval) {
          clearInterval(pollInterval);
          pollInterval = null;
        }
      } catch (err) {
        console.error('Error checking processing status:', err);
      }
    };

    // Initial check
    checkProcessingStatus();

    // Start polling if not already polling
    if (!pollInterval) {
      pollInterval = setInterval(checkProcessingStatus, 2000);
    }

    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [refreshTrigger]);

  const handleReset = async () => {
    setConfirmOpen(false);
    setResetting(true);
    try {
      await api.resetDatabase();
      setRefreshTrigger(prev => prev + 1);
    } catch (err) {
      console.error('Error resetting database:', err);
    } finally {
      setResetting(false);
    }
  };

  const handleDocumentDelete = (docToDelete) => {
    const newSelected = selectedDocuments.filter(
      doc => doc.source_name !== docToDelete.source_name
    );
    setSelectedDocuments(newSelected);
  };

  // Handle document list updates from FileUpload
  const handleUploadStart = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1);
  };

  const handleUploadError = async (filename) => {
    try {
      // Remove the failed document from the backend
      await api.deleteDocument(filename);
    } catch (err) {
      console.error('Error cleaning up failed document:', err);
    } finally {
      // Refresh the document list to remove the failed document from UI
      setRefreshTrigger(prev => prev + 1);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ height: '100vh', py: 4 }}>
      <Typography 
        variant="h4" 
        gutterBottom 
        sx={{ 
          mb: 3,
          fontWeight: 500,
          pl: 3  // Match Grid item padding
        }}
      >
        DocuChat AI
      </Typography>
      <Grid container spacing={3} sx={{ height: 'calc(100% - 80px)' }}>  {/* Adjusted for title */}
        {/* Left Sidebar */}
        <Grid item xs={12} md={3}>
          <Paper
            elevation={3}
            sx={{
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              p: 2
            }}
          >
            <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              {/* Main Content */}
              <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 3 }}>
                <FileUpload 
                  onUploadStart={handleUploadStart}
                  onUploadSuccess={handleUploadSuccess}
                  onUploadError={handleUploadError}
                  isProcessing={processingDocument}
                />
                <DocumentSearch
                  onDocumentsSelect={setSelectedDocuments}
                  selectedDocuments={selectedDocuments}
                  refreshTrigger={refreshTrigger}
                />
              </Box>
              
              {/* Settings Section */}
              <Box 
                sx={{ 
                  mt: 2,
                  pt: 2,
                  borderTop: '1px solid',
                  borderColor: 'divider',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}
              >
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => setSettingsOpen(true)}
                  startIcon={<SettingsIcon />}
                  sx={{ 
                    fontSize: '0.75rem',
                    textTransform: 'none'
                  }}
                >
                  Settings
                </Button>
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  onClick={() => setConfirmOpen(true)}
                  sx={{ 
                    fontSize: '0.75rem',
                    textTransform: 'none'
                  }}
                >
                  {resetting ? 'Resetting...' : 'Reset Database'}
                </Button>
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
                  variant="outlined"
                  size="small"
                  disabled={resetting}
                  sx={{ 
                    fontSize: '0.75rem',
                    textTransform: 'none'
                  }}
                >
                  Reset
                </Button>
              </DialogActions>
            </Dialog>

            {/* Settings Dialog */}
            <Dialog 
              open={settingsOpen} 
              onClose={() => setSettingsOpen(false)}
              maxWidth="md"
              fullWidth
            >
              <DialogTitle>Settings</DialogTitle>
              <DialogContent>
                <SettingsPanel onClose={() => setSettingsOpen(false)} />
              </DialogContent>
            </Dialog>
            </Box>
          </Paper>
        </Grid>

        {/* Main Chat Area */}
        <Grid item xs={12} md={9} sx={{ height: '100%' }}>
          <ChatInterface
            selectedDocuments={selectedDocuments}
            onDocumentDelete={handleDocumentDelete}
          />
        </Grid>
      </Grid>
    </Container>
  );
}

export default App;
