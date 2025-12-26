import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  FlatList,
  Image,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';

// ⚠️ CHANGE THIS IP ADDRESS TO YOUR COMPUTER'S LOCAL IP (e.g., 192.168.1.X) IF USING PHYSICAL DEVICE
// For Android Emulator, use 'http://10.0.2.2:8000'
// For iOS Simulator, use 'http://localhost:8000'
const API_URL = 'http://192.168.1.46:8000';

export default function App() {
  const [messages, setMessages] = useState([
    { id: '1', text: 'Hello! I am talkDOC. specific to medical inquiries only.', sender: 'bot' }
  ]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const flatListRef = useRef(null);

  useEffect(() => {
    // Scroll to bottom when messages change
    setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), 100);
  }, [messages]);

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMsg = { id: Date.now().toString(), text: inputText, sender: 'user' };
    setMessages(prev => [...prev, userMsg]);
    setInputText('');
    setLoading(true);

    try {
      console.log("Sending to:", `${API_URL}/chat`);
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg.text }),
      });

      const data = await response.json();
      console.log("Response:", data);

      const botMsg = { id: (Date.now() + 1).toString(), text: data.response, sender: 'bot' };
      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), text: "Unable to connect to TalkDOC server. Make sure the API is running and IP is correct.", sender: 'bot' }]);
    }
    setLoading(false);
  };

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      alert('Sorry, we need camera roll permissions to make this work!');
      return;
    }

    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false, // Set to false to allow full image
      quality: 0.8,
    });

    if (!result.canceled) {
      const uri = result.assets[0].uri;
      const userMsg = { id: Date.now().toString(), image: uri, sender: 'user' };
      setMessages(prev => [...prev, userMsg]);
      uploadImage(uri);
    }
  };

  const uploadImage = async (uri) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', {
      uri,
      name: 'photo.jpg',
      type: 'image/jpeg',
    });

    try {
      const response = await fetch(`${API_URL}/analyze-image`, {
        method: 'POST',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        body: formData,
      });
      const data = await response.json();
      const botMsg = { id: (Date.now() + 1).toString(), text: data.response, sender: 'bot' };
      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), text: "Error uploading image to server.", sender: 'bot' }]);
    }
    setLoading(false);
  }

  const renderItem = ({ item }) => {
    const isUser = item.sender === 'user';
    return (
      <View style={[styles.messageBubble, isUser ? styles.userBubble : styles.botBubble]}>
        {item.image ? (
          <Image source={{ uri: item.image }} style={styles.uploadedImage} />
        ) : (
          <Text style={styles.messageText}>{item.text}</Text>
        )}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <Text style={styles.headerTitle}>talkDOC 🩺</Text>
      </View>

      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        contentContainerStyle={styles.listContent}
        keyboardShouldPersistTaps="handled"
      />

      {loading && (
        <View style={styles.typingIndicator}>
          <ActivityIndicator size="small" color="#4dabf7" />
          <Text style={styles.typingText}>talkDOC is thinking...</Text>
        </View>
      )}

      <KeyboardAvoidingView
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={Platform.OS === "ios" ? 10 : 0}
      >
        <View style={styles.inputContainer}>
          <TouchableOpacity onPress={pickImage} style={styles.iconButton}>
            <Text style={styles.iconText}>📷</Text>
          </TouchableOpacity>

          <TextInput
            style={styles.input}
            value={inputText}
            onChangeText={setInputText}
            placeholder="Ask a medical question..."
            placeholderTextColor="#888"
          />

          <TouchableOpacity onPress={sendMessage} style={styles.sendButton}>
            <Text style={styles.sendButtonText}>Send</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1b1e', // Dark background
  },
  header: {
    paddingTop: 40,
    paddingBottom: 15,
    backgroundColor: '#1a1b1e',
    borderBottomWidth: 1,
    borderBottomColor: '#2c2e33',
    alignItems: 'center',
  },
  headerTitle: {
    color: '#e7f5ff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  listContent: {
    padding: 15,
    paddingBottom: 20,
  },
  messageBubble: {
    maxWidth: '80%',
    padding: 12,
    borderRadius: 16,
    marginVertical: 5,
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: '#228be6', // Blue
    borderBottomRightRadius: 4,
  },
  botBubble: {
    alignSelf: 'flex-start',
    backgroundColor: '#2c2e33', // Dark Grey
    borderBottomLeftRadius: 4,
  },
  messageText: {
    color: '#fff',
    fontSize: 16,
    lineHeight: 22,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    backgroundColor: '#25262b',
    borderTopWidth: 1,
    borderTopColor: '#2c2e33',
  },
  input: {
    flex: 1,
    backgroundColor: '#1a1b1e',
    color: '#fff',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    marginHorizontal: 10,
    fontSize: 16,
  },
  iconButton: {
    padding: 10,
  },
  iconText: {
    fontSize: 22,
  },
  sendButton: {
    backgroundColor: '#228be6',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
  },
  sendButtonText: {
    color: '#fff',
    fontWeight: 'bold',
  },
  uploadedImage: {
    width: 200,
    height: 200,
    borderRadius: 10,
  },
  typingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    marginLeft: 10,
  },
  typingText: {
    color: '#888',
    marginLeft: 8,
    fontSize: 12,
  },
});
