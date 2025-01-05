import React, { useState, useCallback, useEffect } from 'react';
import PropTypes from 'prop-types';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  Collapse,
  Paper,
  useTheme
} from '@mui/material';
import ExpandLess from '@mui/icons-material/ExpandLess';
import ExpandMore from '@mui/icons-material/ExpandMore';

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

const TableOfContents = ({ toc = [] }) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState({});

  // Log TOC data when it changes
  useEffect(() => {
    console.log('TableOfContents received TOC:', toc);
    if (!toc || toc.length === 0) {
      console.log('No TOC data available');
    } else {
      console.log('TOC structure:', JSON.stringify(toc, null, 2));
    }
  }, [toc]);

  // Memoize expanded state handler
  const toggleExpand = useCallback((index) => {
    setExpanded(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  }, []);

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e, index, hasChildren) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (hasChildren) {
        toggleExpand(index);
      }
    }
  }, [toggleExpand]);

  // Memoize rendering function
  const renderTocItem = useCallback((item, index, level = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expanded[index] ?? true;
    const itemId = `toc-item-${index}`;

    return (
      <React.Fragment key={`${index}-${item.text}`}>
        <ListItem
          button={hasChildren}
          onClick={hasChildren ? () => toggleExpand(index) : undefined}
          onKeyDown={(e) => handleKeyDown(e, index, hasChildren)}
          sx={{
            pl: level * 2,
            py: 0.5,
            minHeight: 36,
            '&:hover': {
              bgcolor: 'action.hover'
            },
            '&:focus-visible': {
              outline: `2px solid ${theme.palette.primary.main}`,
              outlineOffset: '-2px'
            }
          }}
          role="treeitem"
          aria-expanded={hasChildren ? isExpanded : undefined}
          aria-level={level + 1}
          aria-owns={hasChildren ? `${itemId}-group` : undefined}
          id={itemId}
          tabIndex={0}
        >
          <ListItemText
            primary={item.text}
            primaryTypographyProps={{
              variant: 'body2',
              sx: {
                fontWeight: level === 0 ? 600 : 400,
                color: level === 0 ? 'primary.main' : 'text.primary'
              }
            }}
          />
          {hasChildren && (
            <Box 
              component="span" 
              sx={{ ml: 1 }}
              aria-hidden="true"
            >
              {isExpanded ? <ExpandLess /> : <ExpandMore />}
            </Box>
          )}
        </ListItem>
        {hasChildren && (
          <Collapse in={isExpanded} timeout="auto">
            <List 
              component="div" 
              disablePadding
              role="group"
              id={`${itemId}-group`}
              aria-labelledby={itemId}
            >
              {item.children.map((child, childIndex) =>
                renderTocItem(child, `${index}-${childIndex}`, level + 1)
              )}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  }, [expanded, handleKeyDown, theme.palette.primary.main, toggleExpand]);

  if (!toc || toc.length === 0) {
    console.log('Rendering empty TOC state');
    return (
      <Paper
        elevation={0}
        sx={{
          p: 2,
          bgcolor: 'background.default',
          border: '1px dashed',
          borderColor: 'divider'
        }}
      >
        <Typography variant="body2" color="text.secondary" align="center">
          No table of contents available
        </Typography>
      </Paper>
    );
  }

  console.log('Rendering TOC with entries:', toc.length);

  return (
    <Paper
      elevation={3}
      sx={{
        height: '100%',
        overflow: 'auto',
        bgcolor: 'background.paper',
        position: 'relative' // Add position relative for ResizeObserver
      }}
    >
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6" component="h3">
          Table of Contents
        </Typography>
      </Box>
      <List
        component="nav"
        sx={{
          width: '100%',
          bgcolor: 'background.paper',
          p: 1
        }}
      >
        {toc.map((item, index) => renderTocItem(item, index))}
      </List>
    </Paper>
  );
};

TableOfContents.propTypes = {
  toc: PropTypes.arrayOf(
    PropTypes.shape({
      text: PropTypes.string.isRequired,
      children: PropTypes.arrayOf(PropTypes.object)
    })
  )
};

export default TableOfContents;
