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
  Alert,
  Chip,
  Tooltip,
  IconButton,
  InputAdornment
} from '@mui/material';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import ClearIcon from '@mui/icons-material/Clear';
import api from '../api';

const DocumentSearch = ({ onDocumentsSelect, selectedDocuments, refreshTrigger = 0 }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const cachedDocs = useRef([]);

  // Load documents and poll for updates
  useEffect(() => {
    let pollInterval;
    const loadDocuments = async () => {
      try {
        setError(null);
        const docs = await api.getDocumentNames();
        cachedDocs.current = docs;
        setDocuments(docs);
        
        // Check if any documents are still processing
        const hasProcessing = docs.some(doc => doc.status === 'processing');
        if (hasProcessing && !pollInterval) {
          // Start polling if there are processing documents
          pollInterval = setInterval(async () => {
            try {
              const updatedDocs = await api.getDocumentNames();
              cachedDocs.current = updatedDocs;
              setDocuments(updatedDocs);
              
              // Stop polling if no more processing documents
              if (!updatedDocs.some(doc => doc.status === 'processing')) {
                clearInterval(pollInterval);
                pollInterval = null;
              }
            } catch (err) {
              console.error('Error polling documents:', err);
            }
          }, 2000); // Poll every 2 seconds
        }
      } catch (err) {
        setError('Error loading documents');
        console.error('Error loading documents:', err);
      } finally {
        setLoading(false);
      }
    };

    setLoading(true);
    loadDocuments();

    // Cleanup
    return () => {
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
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
        // Merge search results with cached document status
        const mergedResults = results.map(result => {
          const cachedDoc = cachedDocs.current.find(doc => doc.source_name === result.source_name);
          return {
            ...result,
            status: cachedDoc?.status || 'completed',
            chunk_count: cachedDoc?.chunk_count || 0,
            total_chunks: cachedDoc?.total_chunks || 0,
            error: cachedDoc?.error
          };
        });
        setDocuments(mergedResults);
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
    // Only allow selection of completed documents
    if (doc.status !== 'completed') return;

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

  const getStatusChip = (doc) => {
    switch (doc.status) {
      case 'processing':
        return (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CircularProgress size={16} />
            <Typography variant="caption" color="primary">
              Processing...
            </Typography>
          </Box>
        );
      case 'error':
        return (
          <Tooltip title={doc.error || 'Processing error'}>
            <Chip
              icon={<ErrorOutlineIcon />}
              label="Error"
              size="small"
              color="error"
            />
          </Tooltip>
        );
      case 'completed':
        return null; // Don't show any status for completed documents
      default:
        return null;
    }
  };

  return (
    <Box sx={{ 
      display: 'flex',
      flexDirection: 'column',
      flexGrow: 1,
      minHeight: 0  // Important for nested flex containers
    }}>
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
        InputProps={{
          endAdornment: searchQuery ? (
            <InputAdornment position="end">
              <IconButton
                aria-label="clear search"
                onClick={() => setSearchQuery('')}
                edge="end"
                size="small"
                sx={{ 
                  p: 0.5,
                  color: 'rgba(0, 0, 0, 0.38)', // Lighter grey matching placeholder
                  '&:hover': {
                    color: 'rgba(0, 0, 0, 0.54)' // Slightly darker on hover
                  }
                }}
              >
                <ClearIcon sx={{ fontSize: 16 }} />
              </IconButton>
            </InputAdornment>
          ) : null
        }}
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

      <Box 
        sx={{ 
          flexGrow: 1,
          minHeight: 0,  // Important for nested flex containers
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 1,
          bgcolor: 'background.paper',
          display: 'flex',  // Add flex display
          height: 0  // Force container to respect flex sizing
        }}
      >
        <List 
          sx={{ 
            flexGrow: 1,
            overflow: 'auto',
            minWidth: 0,  // Allow list to shrink
            display: 'flex',
            flexDirection: 'column',
            '&::-webkit-scrollbar': {
              width: '8px',
              height: '8px', // For horizontal scrollbar
            },
            '&::-webkit-scrollbar-track': {
              backgroundColor: 'action.hover',
            },
            '&::-webkit-scrollbar-thumb': {
              backgroundColor: 'primary.light',
              borderRadius: '4px',
            },
          }}
        >
          {documents.length === 0 && !loading && (
            <Box sx={{ 
              flexGrow: 1, 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center' 
            }}>
              <Typography color="text.secondary">
                No documents found
              </Typography>
            </Box>
          )}
        {documents.map((doc) => (
          <ListItem
            key={doc.source_name}
            dense
            button
            onClick={() => handleDocumentToggle(doc)}
            disabled={doc.status !== 'completed'}
            sx={{
              opacity: doc.status === 'completed' ? 1 : 0.7,
              '&:hover': {
                backgroundColor: doc.status === 'completed' ? 'action.hover' : 'inherit'
              }
            }}
          >
            <ListItemIcon>
              <Checkbox
                edge="start"
                checked={isSelected(doc)}
                tabIndex={-1}
                disableRipple
                disabled={doc.status !== 'completed'}
              />
            </ListItemIcon>
            <ListItemText
              primary={
                <Tooltip title={doc.title || doc.source_name} placement="top">
                  <Typography
                    variant="body2"
                    sx={{
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      fontSize: '0.875rem'
                    }}
                  >
                    {doc.title || doc.source_name}
                  </Typography>
                </Tooltip>
              }
              secondary={getStatusChip(doc)}
              secondaryTypographyProps={{
                sx: { 
                  mt: 0.25,  // Reduce space between primary and secondary text
                  display: 'block'  // Ensure secondary text is on new line
                }
              }}
            />
          </ListItem>
        ))}
        </List>
      </Box>
    </Box>
  );
};

export default DocumentSearch;
