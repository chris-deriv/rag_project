import React, { useState, useEffect } from 'react';
import {
  TextField,
  List,
  ListItem,
  ListItemText,
  Typography,
  Box,
  CircularProgress,
  Checkbox,
  ListItemIcon,
  Alert
} from '@mui/material';
import api from '../api';

const DocumentSearch = ({ onDocumentsSelect, selectedDocuments, onRefresh, refreshTrigger = 0 }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Initial document load and processing check
  const loadDocuments = async () => {
    try {
      setLoading(true);
      setError(null);
      const docs = await api.getDocumentNames();
      
      // Check if any documents are still processing
      const hasProcessingDocs = docs.some(doc => !doc.chunk_count || doc.chunk_count === 0);
      setIsProcessing(hasProcessingDocs);
      
      setDocuments(docs);
    } catch (err) {
      setError('Error loading documents');
      console.error('Error loading documents:', err);
    } finally {
      setLoading(false);
    }
  };

  // Load documents initially and when refreshTrigger changes
  useEffect(() => {
    if (!searchQuery) { // Only reload if not currently searching
      loadDocuments();
    }
  }, [refreshTrigger]); // Reload when refreshTrigger changes

  // Set up polling only when documents are processing
  useEffect(() => {
    let pollTimeout;
    let pollCount = 0;
    const MAX_POLLS = 30; // 60 seconds total

    const pollDocuments = async () => {
      if (pollCount >= MAX_POLLS) {
        setIsProcessing(false);
        return;
      }

      try {
        const docs = await api.getDocumentNames();
        const hasProcessingDocs = docs.some(doc => !doc.chunk_count || doc.chunk_count === 0);
        
        setDocuments(docs);
        if (onRefresh) {
          onRefresh(docs);
        }

        if (hasProcessingDocs && pollCount < MAX_POLLS) {
          pollCount++;
          pollTimeout = setTimeout(pollDocuments, 2000);
        } else {
          setIsProcessing(false);
        }
      } catch (err) {
        console.error('Error polling documents:', err);
        setIsProcessing(false);
      }
    };

    if (isProcessing) {
      pollTimeout = setTimeout(pollDocuments, 2000);
    }

    return () => {
      if (pollTimeout) {
        clearTimeout(pollTimeout);
      }
    };
  }, [isProcessing, onRefresh]);

  // Handle search
  useEffect(() => {
    let searchTimeout;

    const performSearch = async () => {
      if (!searchQuery.trim()) {
        return; // Don't fetch if search is empty
      }

      try {
        setLoading(true);
        const results = await api.searchTitles(searchQuery);
        setDocuments(results);
      } catch (err) {
        setError('Error searching documents');
        console.error('Error searching documents:', err);
      } finally {
        setLoading(false);
      }
    };

    if (searchQuery) {
      searchTimeout = setTimeout(performSearch, 300);
    }

    return () => {
      if (searchTimeout) {
        clearTimeout(searchTimeout);
      }
    };
  }, [searchQuery]);

  // Handle search input change
  const handleSearchChange = (e) => {
    const value = e.target.value;
    setSearchQuery(value);
    if (!value.trim()) {
      loadDocuments(); // Reload documents when search is cleared
    }
  };

  // Handle document selection
  const handleDocumentToggle = (doc) => {
    if (!doc.chunk_count) return; // Prevent selecting documents that are still processing
    
    const currentIndex = selectedDocuments.findIndex(
      (d) => d.source_name === doc.source_name
    );
    const newSelected = [...selectedDocuments];

    if (currentIndex === -1) {
      newSelected.push(doc);
    } else {
      newSelected.splice(currentIndex, 1);
    }

    onDocumentsSelect(newSelected);
  };

  const isSelected = (doc) => {
    return selectedDocuments.findIndex((d) => d.source_name === doc.source_name) !== -1;
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Documents
      </Typography>

      <TextField
        fullWidth
        size="small"
        label="Search document titles"
        variant="outlined"
        value={searchQuery}
        onChange={handleSearchChange}
        sx={{ mb: 2 }}
      />

      {loading && (
        <Box display="flex" justifyContent="center" my={2}>
          <CircularProgress size={24} />
        </Box>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <List sx={{ maxHeight: 'calc(100vh - 400px)', overflow: 'auto' }}>
        {documents.map((doc, index) => {
          const isProcessing = !doc.chunk_count || doc.chunk_count === 0;
          return (
            <ListItem
              key={index}
              dense
              button
              onClick={() => !isProcessing && handleDocumentToggle(doc)}
              sx={{
                opacity: isProcessing ? 0.7 : 1,
                cursor: isProcessing ? 'default' : 'pointer',
                '&:hover': {
                  backgroundColor: isProcessing ? 'inherit' : undefined
                }
              }}
            >
              <ListItemIcon>
                <Checkbox
                  edge="start"
                  checked={isSelected(doc)}
                  tabIndex={-1}
                  disableRipple
                  disabled={isProcessing}
                />
              </ListItemIcon>
              <ListItemText
                primary={doc.title || doc.source_name}
                secondary={
                  isProcessing ? (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <CircularProgress size={16} />
                      <span>Processing document...</span>
                    </Box>
                  ) : (
                    `${doc.chunk_count} chunks`
                  )
                }
              />
            </ListItem>
          );
        })}
      </List>

      {documents.length === 0 && !loading && (
        <Typography color="textSecondary" align="center">
          No documents found
        </Typography>
      )}
    </Box>
  );
};

export default DocumentSearch;
