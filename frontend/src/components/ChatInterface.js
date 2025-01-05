import React, { useState, useRef, useEffect, useCallback } from 'react';
import TableOfContents from './TableOfContents';
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
  Tab,
  useTheme
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DownloadIcon from '@mui/icons-material/Download';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import api from '../api';

// Create a custom ResizeObserver that ignores errors
const safeResizeObserver = (callback) => {
  try {
    return new ResizeObserver((entries) => {
      // Wrap in requestAnimationFrame to prevent loop limit exceeded error
      window.requestAnimationFrame(() => {
        if (entries && entries.length) {
          callback(entries);
        }
      });
    });
  } catch (e) {
    console.warn('ResizeObserver error:', e);
    return {
      observe: () => {},
      unobserve: () => {},
      disconnect: () => {}
    };
  }
};

const ChatInterface = ({ selectedDocuments, onDocumentDelete }) => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [viewMode, setViewMode] = useState('markdown'); // 'markdown' or 'latex'
  const [currentToc, setCurrentToc] = useState(null);
  const messagesEndRef = useRef(null);
  const theme = useTheme();

  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      const observer = safeResizeObserver(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
      });
      
      observer.observe(messagesEndRef.current);
      
      // Cleanup
      return () => {
        observer.disconnect();
      };
    }
  }, []);

  useEffect(() => {
    const cleanup = scrollToBottom();
    return () => cleanup && cleanup();
  }, [messages, viewMode, scrollToBottom]);

  const handleSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setMessages(prev => [...prev, { type: 'user', content: query }]);
    setLoading(true);
    setError(null);

    try {
      console.log('Making chat API request...');
      const result = await api.chat(
        query,
        selectedDocuments.map(doc => doc.source_name),
        null
      );
      console.log('Chat API response:', result);

      // Ensure consistent content structure
      const messageContent = typeof result.response?.content === 'object' 
        ? result.response.content.content 
        : result.response?.content;

      const tocData = result.response?.table_of_contents;
      console.log('Raw TOC data:', tocData);
      
      const parsedToc = typeof tocData === 'string'
        ? JSON.parse(tocData)
        : tocData;
      console.log('Parsed TOC:', parsedToc);

      setMessages(prev => [...prev, { 
        type: 'assistant', 
        content: messageContent,
        toc: parsedToc
      }]);
      
      // Update current TOC if available
      if (parsedToc) {
        console.log('Setting current TOC:', parsedToc);
        setCurrentToc(parsedToc);
      } else {
        console.log('No TOC data available in response');
      }
    } catch (err) {
      // Extract error details
      const errorMessage = err.response?.data?.error || 'Error getting response';
      const errorDetails = err.response?.data?.details || err.message;
      const errorCode = err.response?.status;

      // Create user-friendly error message
      const userMessage = errorCode 
        ? `Error (${errorCode}): ${errorMessage}`
        : errorMessage;

      // Log error for debugging
      console.error('Chat error:', {
        message: errorMessage,
        details: errorDetails,
        code: errorCode,
        error: err
      });

      setError(userMessage);
      setMessages(prev => [...prev, { 
        type: 'error', 
        content: userMessage,
        details: errorDetails
      }]);

      // Clear error after 5 seconds
      setTimeout(() => setError(null), 5000);
    } finally {
      setLoading(false);
      setQuery('');
    }
  }, [query, selectedDocuments]);

  // Add keyboard shortcut handler
  useEffect(() => {
    const handleKeyPress = (e) => {
      // Ctrl/Cmd + Enter to submit
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (!loading && query.trim() && selectedDocuments.length > 0) {
          handleSubmit(e);
        }
      }
      // Esc to clear input
      if (e.key === 'Escape') {
        setQuery('');
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [loading, query, selectedDocuments.length, handleSubmit]);

  const escapeLatex = (text) => {
    if (!text) return '';  // Return empty string for undefined/null input
    return text
      .replace(/\\/g, '\\textbackslash{}')
      .replace(/[&%$#_{}]/g, '\\$&')
      .replace(/\^/g, '\\textasciicircum{}')
      .replace(/~/g, '\\textasciitilde{}');
  };

  const markdownToLatex = (markdown) => {
    if (!markdown) return '';  // Return empty string for undefined/null input
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
        `\\item ${item.replace(/^[ ]*[-*+][ ]/, '')}`
      ).join('\n');
      return `\\begin{itemize}${items}\\end{itemize}`;
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
    if (!content) return null;  // Return null for undefined/null input
    
    // Split content by math delimiters
    const parts = content.split(/(\$\$[\s\S]*?\$\$|\$[^$]*\$)/g);
    
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
        // Regular LaTeX content - style it like a rendered document
        const processedText = markdownToLatex(part);
        return (
          <Paper
            key={index}
            elevation={1}
            sx={{
              padding: '1rem 1rem',
              color: '#000',
              backgroundColor: '#f8f8f8',
              fontFamily: 'Arial, sans-serif',
              fontSize: '1rem',
              lineHeight: '1.6',
              textAlign: 'justify',
              maxWidth: '100%',
              margin: '1rem 0',
              boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
              '& h2': {
                fontFamily: 'Arial, sans-serif',
                color: '#000',
                borderBottom: '1px solid #ddd',
                paddingBottom: '0.25rem',
                marginBottom: '1rem',
                fontSize: '1.75rem'
              },
              '& h3': {
                fontFamily: 'Arial, sans-serif',
                color: '#000',
                marginTop: '0.25rem',
                marginBottom: '0.25rem',
                fontSize: '1.5rem'
              },
              '& ul': {
                marginTop: 0,
                marginLeft: '1.5rem',
                paddingLeft: '1rem',
                marginBottom: '0.5rem'
              },
              '& li': {
                marginBottom: '0.25rem',
                position: 'relative'
              },
              '& strong': {
                fontWeight: 600,
                color: '#000'
              },
              '& em': {
                fontStyle: 'italic',
                color: '#000'
              }
            }}
            component="div"
            dangerouslySetInnerHTML={{
              __html: processedText
                // Basic LaTeX to HTML conversion for common elements
                .replace(/\\textbf{([^}]+)}/g, '<strong>$1</strong>')
                .replace(/\\textit{([^}]+)}/g, '<em>$1</em>')
                .replace(/\\begin{itemize}/g, '<ul style="margin: 0.25rem; padding-left: 1.5rem;">')
                .replace(/\\end{itemize}/g, '</ul>')
                .replace(/\\item\s*/g, '<li style="margin-bottom: 0.25rem;">')
                .replace(/\n\n/g, '<div style="margin: 0.75rem 0;"></div>')
                .replace(/\\section\*{([^}]+)}/g, '<h2 style="margin: 0.5rem 0;">$1</h2>')
                .replace(/\\subsection\*{([^}]+)}/g, '<h3 style="margin: 0.25rem 0;">$1</h3>')
            }}
          />
        );
      }
    });
  };

  const convertToLatex = (singleMessage = null) => {
    if (singleMessage) {
      // For single message export, only include the message content
      return `\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{amssymb}
\\usepackage{graphicx}
\\usepackage{listings}
\\usepackage{hyperref}
\\geometry{margin=1in}

\\lstset{
  basicstyle=\\ttfamily\\small,
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\\tiny,
  showstringspaces=false
}

\\begin{document}

${markdownToLatex(singleMessage.content)}

\\end{document}`;
    } else {
      // For full conversation export
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

      // Convert all messages to LaTeX format
      if (messages && messages.length > 0) {
        messages.forEach(message => {
          if (!message || !message.content) return;  // Skip invalid messages
          
          const roleColor = message.type === 'user' ? 'userColor' : 
                           message.type === 'error' ? 'errorColor' : 
                           'assistantColor';
          
          const role = (message.type || 'unknown').charAt(0).toUpperCase() + 
                      (message.type || 'unknown').slice(1);
          
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
      }

      latexContent += "\\end{document}";
      return latexContent;
    }
  };

  const handleExport = (message = null) => {
    // Log export request details
    console.log('Export requested:', { 
      type: message ? 'single_message' : 'full_conversation',
      messageType: message?.type,
      messagesCount: messages?.length,
      validMessages: messages?.filter(m => m.content)?.length
    });

    // For single message export
    if (message) {
      if (!message.content) {
        console.log('Skipping export: Single message has no content');
        return;
      }
      if (message.type !== 'assistant') {
        console.log('Skipping export: Not an assistant message');
        return;
      }
      console.log('Proceeding with single message export');
    }
    // For full conversation export
    else {
      const validMessages = messages?.filter(m => m.content)?.length || 0;
      if (validMessages === 0) {
        console.log('Skipping export: No valid messages in conversation');
        return;
      }
      console.log('Proceeding with full conversation export');
    }

    try {
      console.log('Creating LaTeX content');
      const latexContent = convertToLatex(message);
      console.log('Creating download link');
      const blob = new Blob([latexContent], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = message ? 'response.tex' : 'chat_conversation.tex';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      console.log('Export completed');
    } catch (error) {
      console.error('Export failed:', error);
    }
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

  const renderMessage = (message) => {
    // Handle case where message.content is an object with content property
    const content = typeof message.content === 'object' && message.content.content 
      ? message.content.content 
      : message.content;

    if (viewMode === 'latex' && message.type === 'assistant') {
      return renderLatexMessage(content);
    }
    return content;
  };

  return (
    <Box sx={{ display: 'flex', gap: 2, height: '100%' }}>
      <Paper elevation={3} sx={{ p: 3, height: '100%', display: 'flex', flexDirection: 'column', flex: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h2">
          Chat
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', bgcolor: 'background.paper', borderRadius: 1 }}>
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
                  py: 1,
                  mr: 0
                }}
              />
            </Tabs>
            {messages.length > 0 && viewMode === 'latex' && (
              <Tooltip title="Export as LaTeX Document">
                <IconButton 
                  onClick={(e) => {
                    e.stopPropagation();
                    handleExport(null);
                  }} 
                  color="primary"
                  size="small"
                  sx={{
                    ml: 1,
                    mr: 0.5,
                    '&:hover': {
                      backgroundColor: 'rgba(0, 0, 0, 0.04)',
                    }
                  }}
                >
                  <DownloadIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </Box>
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
          bgcolor: viewMode === 'latex' ? '#fff' : 'background.default',
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
                maxWidth: message.type === 'user' ? '80%' : '100%',
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
                borderColor: message.type === 'error' ? 'error.main' : 'divider',
                width: message.type === 'user' ? 'auto' : '100%'
              }}
            >
              <Box sx={{ position: 'relative', width: '100%' }}>
                <Typography
                  variant="body1"
                  component="div"
                  sx={{ whiteSpace: 'pre-wrap' }}
                >
                  {renderMessage(message)}
                </Typography>
                {message.type === 'assistant' && viewMode === 'latex' && (
                  <Tooltip title="Export Message as LaTeX">
                    <IconButton
                      onClick={(e) => {
                        e.stopPropagation();
                        handleExport(message);
                      }}
                      color="primary"
                      size="small"
                      sx={{
                        position: 'absolute',
                        top: '-0.5rem',
                        right: '-0.5rem',
                        backgroundColor: 'background.paper',
                        boxShadow: 1,
                        '&:hover': {
                          backgroundColor: 'background.paper',
                        }
                      }}
                    >
                      <DownloadIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                )}
              </Box>
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
              : "Type your message... (Ctrl+Enter to send, Esc to clear)"}
            variant="outlined"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading || selectedDocuments.length === 0}
            size="small"
            multiline
            maxRows={4}
            aria-label="Chat input"
            aria-describedby="chat-input-help"
            inputProps={{
              'data-lpignore': 'true'
            }}
            InputProps={{
              'aria-controls': 'chat-messages',
              'aria-expanded': 'true',
              role: 'textbox',
              'aria-multiline': 'true'
            }}
          />
          <Button
            type="submit"
            variant="contained"
            endIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SendIcon />}
            disabled={loading || !query.trim() || selectedDocuments.length === 0}
            aria-label={loading ? "Sending message..." : "Send message"}
            title="Send message (Ctrl+Enter)"
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
      
      <Box sx={{ width: 300, height: '100%' }}>
        <TableOfContents toc={currentToc} />
      </Box>
    </Box>
  );
};

export default ChatInterface;
