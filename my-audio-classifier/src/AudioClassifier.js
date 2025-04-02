// src/AudioClassifier.js
import React, { useState } from 'react';
import axios from 'axios';
import { Container, Row, Col, Card, Form, Button, Spinner } from 'react-bootstrap';

function AudioClassifier() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = event => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async event => {
    event.preventDefault();
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:5000/classify', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (error) {
      console.error("Error classifying audio:", error);
    }
    setLoading(false);
  };

  return (
    <Container className="mt-5">
      <Row className="justify-content-md-center">
        <Col md="6">
          <Card>
            <Card.Header as="h4">Audio Classification</Card.Header>
            <Card.Body>
              <Form onSubmit={handleSubmit}>
                <Form.Group className="mb-3" controlId="audioFile">
                  <Form.Label>Select an Audio File</Form.Label>
                  <Form.Control type="file" accept="audio/*" onChange={handleFileChange} />
                </Form.Group>
                <Button variant="primary" type="submit" disabled={loading}>
                  {loading ? <Spinner animation="border" size="sm" /> : 'Classify Audio'}
                </Button>
              </Form>
              {result && (
                <Card className="mt-4">
                  <Card.Body>
                    <Card.Title>Classification Result</Card.Title>
                    <Card.Text>
                      <strong>Predicted Label Index:</strong> {result.predicted_label}<br />
                      <strong>Class:</strong> {result.class}
                    </Card.Text>
                  </Card.Body>
                </Card>
              )}
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default AudioClassifier;
