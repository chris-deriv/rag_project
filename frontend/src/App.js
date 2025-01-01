import React, { useState } from 'react';
import api from './api';
import { Container, Grid, Paper, Box, Button, Dialog, DialogTitle, DialogContent, DialogActions, Typography } from '@mui/material';
import FileUpload from './components/FileUpload';
import DocumentSearch from './components/DocumentSearch';
import ChatInterface from './components/ChatInterface';

function App() {
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [resetting, setResetting] = useState(false);

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
  const handleUploadSuccess = () => {
    setRefreshTrigger(prev => prev + 1); // Only increment refresh trigger
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
                <FileUpload onUploadSuccess={handleUploadSuccess} />
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
                  justifyContent: 'flex-end'
                }}
              >
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
