import React, { useState, useEffect, useCallback, useRef } from 'react';
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

const DocumentSearch = ({ onDocumentsSelect, selectedDocuments, refreshTrigger = 0 }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const cachedDocs = useRef([]);

  // Load documents only when refreshTrigger changes
  useEffect(() => {
    const loadDocuments = async () => {
      try {
        setLoading(true);
        setError(null);
        const docs = await api.getDocumentNames();
        
        // Only show completed documents (those with chunks)
        const completedDocs = docs.filter(doc => doc.chunk_count > 0);
        cachedDocs.current = completedDocs;
        setDocuments(completedDocs);
      } catch (err) {
        setError('Error loading documents');
        console.error('Error loading documents:', err);
      } finally {
        setLoading(false);
      }
    };

    loadDocuments();
  }, [refreshTrigger]);

  // Handle search
  useEffect(() => {
    if (!searchQuery.trim()) {
      setDocuments(cachedDocs.current);
      return;
    }

    let timeoutId;
    const performSearch = async () => {
      try {
        setLoading(true);
        const results = await api.searchTitles(searchQuery);
        
        // Only show completed documents in search results
        const completedResults = results.filter(doc => doc.chunk_count > 0);
        setDocuments(completedResults);
      } catch (err) {
        setError('Error searching documents');
        console.error('Error searching documents:', err);
      } finally {
        setLoading(false);
      }
    };

    timeoutId = setTimeout(performSearch, 300);

    return () => {
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [searchQuery]);

  // Handle document selection
  const handleDocumentToggle = useCallback((doc) => {
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
  }, [selectedDocuments, onDocumentsSelect]);

  const isSelected = useCallback((doc) => {
    return selectedDocuments.findIndex((d) => d.source_name === doc.source_name) !== -1;
  }, [selectedDocuments]);

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
        onChange={(e) => setSearchQuery(e.target.value)}
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
        {documents.map((doc) => (
          <ListItem
            key={doc.source_name}
            dense
            button
            onClick={() => handleDocumentToggle(doc)}
          >
            <ListItemIcon>
              <Checkbox
                edge="start"
                checked={isSelected(doc)}
                tabIndex={-1}
                disableRipple
              />
            </ListItemIcon>
            <ListItemText
              primary={doc.title || doc.source_name}
              secondary={`${doc.chunk_count} chunks`}
            />
          </ListItem>
        ))}
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
