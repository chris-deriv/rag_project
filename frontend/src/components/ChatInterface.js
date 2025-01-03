import React, { useState, useRef, useEffect } from 'react';
import {
  TextField,
  Button,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Stack,
  Chip,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DownloadIcon from '@mui/icons-material/Download';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
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

    setMessages(prev => [...prev, { type: 'user', content: query }]);
    setLoading(true);
    setError(null);

    try {
      const result = await api.chat(
        query,
        selectedDocuments.map(doc => doc.source_name),
        null
      );
      setMessages(prev => [...prev, { type: 'assistant', content: result.response }]);
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Error getting response';
      setError(errorMessage);
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: errorMessage,
        details: err.response?.data?.details
      }]);
    } finally {
      setLoading(false);
      setQuery('');
    }
  };

  const escapeLatex = (text) => {
    return text
      .replace(/\\/g, '\\textbackslash{}')
      .replace(/[&%$#_{}]/g, '\\$&')
      .replace(/\^/g, '\\textasciicircum{}')
      .replace(/~/g, '\\textasciitilde{}');
  };

  const convertToLatex = () => {
    // LaTeX document preamble
    let latexContent = `\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{xcolor}
\\usepackage{geometry}
\\usepackage{listings}
\\geometry{margin=1in}

\\definecolor{userColor}{RGB}{25,118,210}
\\definecolor{assistantColor}{RGB}{66,66,66}
\\definecolor{errorColor}{RGB}{211,47,47}

\\begin{document}
\\section*{Chat Conversation}

`;

    // Add selected documents
    if (selectedDocuments.length > 0) {
      latexContent += "\\subsection*{Selected Documents}\n\\begin{itemize}\n";
      selectedDocuments.forEach(doc => {
        latexContent += `\\item ${escapeLatex(doc.source_name)}\n`;
      });
      latexContent += "\\end{itemize}\n\\vspace{1em}\n";
    }

    // Convert messages to LaTeX format
    messages.forEach(message => {
      const roleColor = message.type === 'user' ? 'userColor' : 
                       message.type === 'error' ? 'errorColor' : 
                       'assistantColor';
      
      const role = message.type.charAt(0).toUpperCase() + message.type.slice(1);
      
      latexContent += `\\begin{quote}
\\textcolor{${roleColor}}{\\textbf{${role}:}}

${escapeLatex(message.content)}
\\end{quote}

`;

      if (message.details) {
        latexContent += `\\begin{quote}
\\textcolor{errorColor}{\\small ${escapeLatex(message.details)}}
\\end{quote}

`;
      }
    });

    latexContent += "\\end{document}";
    return latexContent;
  };

  const handleExport = () => {
    const latexContent = convertToLatex();
    const blob = new Blob([latexContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'chat_conversation.tex';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const getNoDocumentsMessage = () => {
    if (selectedDocuments.length === 0) {
      return (
        <Alert severity="info" sx={{ mb: 2 }}>
          Please select one or more completed documents to start chatting
        </Alert>
      );
    }
    return null;
  };

  return (
    <Paper elevation={3} sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h2">
          Chat
        </Typography>
        {messages.length > 0 && (
          <Tooltip title="Export as LaTeX">
            <IconButton onClick={handleExport} color="primary">
              <DownloadIcon />
            </IconButton>
          </Tooltip>
        )}
      </Box>

      {getNoDocumentsMessage()}

      {selectedDocuments && selectedDocuments.length > 0 && (
        <Box mb={2}>
          <Stack direction="row" spacing={1} flexWrap="wrap" gap={1}>
            {selectedDocuments.map((doc, index) => (
              <Chip
                key={index}
                label={doc.source_name}
                color="primary"
                variant="outlined"
                onDelete={() => onDocumentDelete(doc)}
                icon={<CheckCircleOutlineIcon />}
              />
            ))}
          </Stack>
        </Box>
      )}

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
                bgcolor: message.type === 'user' 
                  ? 'primary.main' 
                  : message.type === 'error' 
                    ? 'error.light'
                    : 'background.paper',
                color: message.type === 'user' 
                  ? 'primary.contrastText' 
                  : message.type === 'error'
                    ? 'error.contrastText'
                    : 'text.primary',
                border: message.type !== 'user' ? '1px solid' : 'none',
                borderColor: message.type === 'error' ? 'error.main' : 'divider'
              }}
            >
              <Typography
                variant="body1"
                component="div"
                sx={{ whiteSpace: 'pre-wrap' }}
              >
                {message.content}
              </Typography>
              {message.details && (
                <Typography
                  variant="caption"
                  color="error"
                  sx={{ mt: 1, display: 'block' }}
                >
                  {message.details}
                </Typography>
              )}
            </Box>
          </Box>
        ))}
        <div ref={messagesEndRef} />
      </Box>

      <form onSubmit={handleSubmit}>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            placeholder={selectedDocuments.length === 0 
              ? "Select documents to start chatting..." 
              : "Type your message..."}
            variant="outlined"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading || selectedDocuments.length === 0}
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
