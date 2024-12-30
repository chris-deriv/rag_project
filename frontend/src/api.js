import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

const api = {
  // Upload a document
  uploadDocument: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  },

  // Search document titles
  searchTitles: async (query) => {
    const response = await axios.get(`${API_BASE_URL}/search-titles`, {
      params: { q: query }
    });
    return response.data;
  },

  // Get all document names
  getDocumentNames: async () => {
    const response = await axios.get(`${API_BASE_URL}/document-names`);
    return response.data;
  },

  // Get document chunks
  getDocumentChunks: async (sourceName) => {
    const response = await axios.get(`${API_BASE_URL}/document-chunks/${sourceName}`);
    return response.data;
  },

  // Chat/Query documents
  chat: async (query, source_names = [], title = null) => {
    const response = await axios.post(`${API_BASE_URL}/chat`, {
      query,
      source_names, // Now expects an array of source names
      title
    });
    return response.data;
  }
};

export default api;
