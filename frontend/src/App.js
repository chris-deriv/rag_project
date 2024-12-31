import React, { useState } from 'react';
import { Container, Grid, Paper } from '@mui/material';
import FileUpload from './components/FileUpload';
import DocumentSearch from './components/DocumentSearch';
import ChatInterface from './components/ChatInterface';

function App() {
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [documentList, setDocumentList] = useState([]);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleDocumentDelete = (docToDelete) => {
    const newSelected = selectedDocuments.filter(
      doc => doc.source_name !== docToDelete.source_name
    );
    setSelectedDocuments(newSelected);
  };

  // Handle document list updates
  const handleDocumentListUpdate = (docs) => {
    setDocumentList(docs);
    setRefreshTrigger(prev => prev + 1); // Increment refresh trigger
    // Remove any selected documents that are no longer in the list
    setSelectedDocuments(prev => 
      prev.filter(selected => 
        docs.some(doc => doc.source_name === selected.source_name)
      )
    );
  };

  return (
    <Container maxWidth="xl" sx={{ height: '100vh', py: 4 }}>
      <Grid container spacing={3} sx={{ height: 'calc(100% - 32px)' }}>
        {/* Left Sidebar */}
        <Grid item xs={12} md={3}>
          <Paper
            elevation={3}
            sx={{
              height: '100%',
              overflow: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: 3,
              p: 2
            }}
          >
            <FileUpload onUploadSuccess={handleDocumentListUpdate} />
            <DocumentSearch
              onDocumentsSelect={setSelectedDocuments}
              selectedDocuments={selectedDocuments}
              onRefresh={handleDocumentListUpdate}
              refreshTrigger={refreshTrigger}
            />
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
