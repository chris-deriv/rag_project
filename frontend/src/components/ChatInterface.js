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
  Tooltip,
  Tabs,
  Tab
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DownloadIcon from '@mui/icons-material/Download';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import api from '../api';

const ChatInterface = ({ selectedDocuments, onDocumentDelete }) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('markdown'); // 'markdown' or 'latex'
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, viewMode]);

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

  const markdownToLatex = (markdown) => {
    let latex = markdown;
    
    // Convert markdown code blocks to LaTeX listings
    latex = latex.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
      return `\\begin{lstlisting}[language=${lang || 'text'}]\n${code}\n\\end{lstlisting}`;
    });

    // Convert markdown inline code to LaTeX texttt
    latex = latex.replace(/`([^`]+)`/g, '\\texttt{$1}');

    // Convert markdown bold to LaTeX textbf
    latex = latex.replace(/\*\*([^*]+)\*\*/g, '\\textbf{$1}');

    // Convert markdown italic to LaTeX textit
    latex = latex.replace(/\*([^*]+)\*/g, '\\textit{$1}');

    // Convert markdown lists to LaTeX itemize
    latex = latex.replace(/(?:^|\n)((?:[ ]*[-*+][ ].+\n?)+)/g, (_, list) => {
      const items = list.trim().split('\n').map(item => 
        `  \\item ${item.replace(/^[ ]*[-*+][ ]/, '')}`
      ).join('\n');
      return `\\begin{itemize}\n${items}\n\\end{itemize}\n`;
    });

    // Convert markdown headers to LaTeX sections
    latex = latex.replace(/^#{1,6}\s+(.+)$/gm, (match, title) => {
      const level = match.trim().indexOf(' ');
      const sections = ['section', 'subsection', 'subsubsection', 'paragraph', 'subparagraph'];
      return `\\${sections[Math.min(level - 1, sections.length - 1)]}*{${title}}`;
    });

    return latex;
  };

  const renderLatexMessage = (content) => {
    // Split content by math delimiters
    const parts = content.split(/(\$\$[\s\S]*?\$\$|\$[^\$]*\$)/g);
    
    return parts.map((part, index) => {
      if (part.startsWith('$$') && part.endsWith('$$')) {
        // Display math
        const math = part.slice(2, -2);
        try {
          return <BlockMath key={index} math={math} />;
        } catch (e) {
          return <pre key={index}>{math}</pre>;
        }
      } else if (part.startsWith('$') && part.endsWith('$')) {
        // Inline math
        const math = part.slice(1, -1);
        try {
          return <InlineMath key={index} math={math} />;
        } catch (e) {
          return <span key={index}>{math}</span>;
        }
      } else {
        // Regular text - convert markdown except math
        const processedText = markdownToLatex(part);
        return <span key={index}>{processedText}</span>;
      }
    });
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
\\usepackage{hyperref}
\\geometry{margin=1in}

\\definecolor{userColor}{RGB}{25,118,210}
\\definecolor{assistantColor}{RGB}{66,66,66}
\\definecolor{errorColor}{RGB}{211,47,47}

\\lstset{
  basicstyle=\\ttfamily\\small,
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\\tiny,
  showstringspaces=false
}

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

${markdownToLatex(message.content)}
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
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Tabs
            value={viewMode}
            onChange={(_, newValue) => setViewMode(newValue)}
            sx={{ minHeight: 0 }}
          >
            <Tab 
              label="Markdown" 
              value="markdown"
              sx={{ 
                minHeight: 0,
                py: 1
              }}
            />
            <Tab 
              label="LaTeX Preview" 
              value="latex"
              sx={{ 
                minHeight: 0,
                py: 1
              }}
            />
          </Tabs>
          {messages.length > 0 && (
            <Tooltip title="Export as LaTeX">
              <IconButton onClick={handleExport} color="primary">
                <DownloadIcon />
              </IconButton>
            </Tooltip>
          )}
        </Box>
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
                {viewMode === 'markdown' ? 
                  message.content : 
                  renderLatexMessage(message.content)}
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
