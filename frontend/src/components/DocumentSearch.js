import React, { useState, useEffect } from 'react';
import {
  TextField,
  List,
  ListItem,
  ListItemText,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Checkbox,
  ListItemIcon
} from '@mui/material';
import api from '../api';

const DocumentSearch = ({ onDocumentsSelect, selectedDocuments }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load all documents initially
  useEffect(() => {
    const loadDocuments = async () => {
      try {
        setLoading(true);
        const docs = await api.getDocumentNames();
        setDocuments(docs);
      } catch (err) {
        setError('Error loading documents');
      } finally {
        setLoading(false);
      }
    };

    loadDocuments();
  }, []);

  // Search documents when query changes
  useEffect(() => {
    const searchDocuments = async () => {
      if (!searchQuery.trim()) {
        const docs = await api.getDocumentNames();
        setDocuments(docs);
        return;
      }

      try {
        setLoading(true);
        const results = await api.searchTitles(searchQuery);
        setDocuments(results);
      } catch (err) {
        setError('Error searching documents');
      } finally {
        setLoading(false);
      }
    };

    const debounce = setTimeout(searchDocuments, 300);
    return () => clearTimeout(debounce);
  }, [searchQuery]);

  // Handle document selection
  const handleDocumentToggle = (doc) => {
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
        onChange={(e) => setSearchQuery(e.target.value)}
        sx={{ mb: 2 }}
      />

      {loading && (
        <Box display="flex" justifyContent="center" my={2}>
          <CircularProgress size={24} />
        </Box>
      )}

      {error && (
        <Typography color="error" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}

      <List sx={{ maxHeight: 'calc(100vh - 400px)', overflow: 'auto' }}>
        {documents.map((doc, index) => (
          <ListItem
            key={index}
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
              secondary={
                doc.chunk_count
                  ? `${doc.chunk_count} chunks`
                  : 'Processing...'
              }
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
