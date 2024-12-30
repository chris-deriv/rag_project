import React, { useState, useRef, useEffect } from 'react';
import {
  TextField,
  Button,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Stack,
  Chip
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import api from '../api';

const ChatInterface = ({ selectedDocuments, onDocumentDelete }) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    // Add user message
    setMessages(prev => [...prev, { type: 'user', content: query }]);
    setLoading(true);
    setError(null);

    try {
      const result = await api.chat(
        query,
        selectedDocuments.map(doc => doc.source_name),
        null // title is not needed when using multiple documents
      );
      // Add assistant message
      setMessages(prev => [...prev, { type: 'assistant', content: result.response }]);
    } catch (err) {
      setError(err.response?.data?.error || 'Error getting response');
      // Add error message
      setMessages(prev => [...prev, { type: 'error', content: err.response?.data?.error || 'Error getting response' }]);
    } finally {
      setLoading(false);
      setQuery('');
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Typography variant="h6" component="h2" gutterBottom>
        Chat
      </Typography>

      {selectedDocuments && selectedDocuments.length > 0 && (
        <Box mb={2}>
          <Stack direction="row" spacing={1} flexWrap="wrap" gap={1}>
            {selectedDocuments.map((doc, index) => (
              <Chip
                key={index}
                label={doc.title || doc.source_name}
                color="primary"
                variant="outlined"
                onDelete={() => onDocumentDelete(doc)}
              />
            ))}
          </Stack>
        </Box>
      )}

      {/* Chat messages area */}
      <Box
        sx={{
          flex: 1,
          mb: 2,
          overflowY: 'auto',
          bgcolor: 'background.default',
          borderRadius: 1,
          p: 2,
          border: '1px solid',
          borderColor: 'divider'
        }}
      >
        {messages.map((message, index) => (
          <Box
            key={index}
            sx={{
              mb: 2,
              display: 'flex',
              flexDirection: 'column',
              alignItems: message.type === 'user' ? 'flex-end' : 'flex-start'
            }}
          >
            <Box
              sx={{
                maxWidth: '80%',
                p: 2,
                borderRadius: 2,
                bgcolor: message.type === 'user' ? 'primary.main' : 'background.paper',
                color: message.type === 'user' ? 'primary.contrastText' : 'text.primary',
                border: message.type !== 'user' ? '1px solid' : 'none',
                borderColor: 'divider'
              }}
            >
              <Typography
                variant="body1"
                component="div"
                sx={{ whiteSpace: 'pre-wrap' }}
              >
                {message.content}
              </Typography>
            </Box>
          </Box>
        ))}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input area */}
      <form onSubmit={handleSubmit}>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            placeholder="Type your message..."
            variant="outlined"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
            size="small"
          />
          <Button
            type="submit"
            variant="contained"
            endIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
            disabled={loading || !query.trim() || selectedDocuments.length === 0}
          >
            Send
          </Button>
        </Box>
      </form>

      {error && (
        <Typography color="error" sx={{ mt: 2 }}>
          {error}
        </Typography>
      )}
    </Paper>
  );
};

export default ChatInterface;
