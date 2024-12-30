import React, { useState } from 'react';
import { Container, Grid, Paper } from '@mui/material';
import FileUpload from './components/FileUpload';
import DocumentSearch from './components/DocumentSearch';
import ChatInterface from './components/ChatInterface';

function App() {
  const [selectedDocuments, setSelectedDocuments] = useState([]);

  const handleDocumentDelete = (docToDelete) => {
    const newSelected = selectedDocuments.filter(
      doc => doc.source_name !== docToDelete.source_name
    );
    setSelectedDocuments(newSelected);
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
            <FileUpload />
            <DocumentSearch
              onDocumentsSelect={setSelectedDocuments}
              selectedDocuments={selectedDocuments}
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
