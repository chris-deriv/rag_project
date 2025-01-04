import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemText,
  Collapse,
  Paper
} from '@mui/material';
import ExpandLess from '@mui/icons-material/ExpandLess';
import ExpandMore from '@mui/icons-material/ExpandMore';

const TableOfContents = ({ toc }) => {
  const [expanded, setExpanded] = React.useState({});

  const toggleExpand = (index) => {
    setExpanded(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const renderTocItem = (item, index, level = 0) => {
    const hasChildren = item.children && item.children.length > 0;
    const isExpanded = expanded[index] ?? true;

    return (
      <React.Fragment key={`${index}-${item.text}`}>
        <ListItem
          button={hasChildren}
          onClick={hasChildren ? () => toggleExpand(index) : undefined}
          sx={{
            pl: level * 2,
            py: 0.5,
            minHeight: 36,
            '&:hover': {
              bgcolor: 'action.hover'
            }
          }}
        >
          <ListItemText
            primary={item.text}
            primaryTypographyProps={{
              variant: 'body2',
              sx: {
                fontWeight: level === 0 ? 600 : 400,
                color: level === 0 ? 'primary.main' : 'text.primary'
              }}
            }
          />
          {hasChildren && (
            <Box component="span" sx={{ ml: 1 }}>
              {isExpanded ? <ExpandLess /> : <ExpandMore />}
            </Box>
          )}
        </ListItem>
        {hasChildren && (
          <Collapse in={isExpanded} timeout="auto">
            <List component="div" disablePadding>
              {item.children.map((child, childIndex) =>
                renderTocItem(child, `${index}-${childIndex}`, level + 1)
              )}
            </List>
          </Collapse>
        )}
      </React.Fragment>
    );
  };

  if (!toc || toc.length === 0) {
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

  return (
    <Paper
      elevation={3}
      sx={{
        height: '100%',
        overflow: 'auto',
        bgcolor: 'background.paper'
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

export default TableOfContents;
