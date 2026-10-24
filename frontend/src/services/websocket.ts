import { PredictionEvent } from '../types';

export type ConnectionStatus = 'DISCONNECTED' | 'CONNECTING' | 'CONNECTED' | 'ERROR';

export class SignBridgeWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private onMessageCallback: (data: PredictionEvent) => void;
  private onStatusChangeCallback: (status: ConnectionStatus) => void;
  private reconnectTimeout: number | null = null;
  private isExplicitlyClosed = false;
  private isSendingFrame = false;

  constructor(
    onMessage: (data: PredictionEvent) => void,
    onStatusChange: (status: ConnectionStatus) => void
  ) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // If running in dev mode via Vite proxy or direct port 8000
    const host = window.location.port === '5173' ? '127.0.0.1:8000' : window.location.host;
    this.url = `${protocol}//${host}/ws/predict`;
    this.onMessageCallback = onMessage;
    this.onStatusChangeCallback = onStatusChange;
  }

  public connect() {
    this.isExplicitlyClosed = false;
    this.onStatusChangeCallback('CONNECTING');

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.onStatusChangeCallback('CONNECTED');
        if (this.reconnectTimeout) {
          clearTimeout(this.reconnectTimeout);
          this.reconnectTimeout = null;
        }
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === 'prediction') {
            this.onMessageCallback(payload as PredictionEvent);
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e);
        } finally {
          this.isSendingFrame = false;
        }
      };

      this.ws.onerror = (err) => {
        console.warn('WebSocket encountered error:', err);
        this.onStatusChangeCallback('ERROR');
        this.isSendingFrame = false;
      };

      this.ws.onclose = () => {
        this.onStatusChangeCallback('DISCONNECTED');
        this.isSendingFrame = false;
        if (!this.isExplicitlyClosed) {
          this.scheduleReconnect();
        }
      };
    } catch (e) {
      console.error('Failed to initiate WebSocket connection:', e);
      this.onStatusChangeCallback('ERROR');
      this.scheduleReconnect();
    }
  }

  public sendFrame(base64Data: string): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    // Drop frame if previous frame response hasn't returned yet (prevents buffer backlog)
    if (this.isSendingFrame) {
      return false;
    }

    try {
      this.isSendingFrame = true;
      this.ws.send(JSON.stringify({
        type: 'frame',
        image: base64Data,
        timestamp: Date.now() / 1000,
      }));
      return true;
    } catch (e) {
      console.error('Failed to dispatch frame over WebSocket:', e);
      this.isSendingFrame = false;
      return false;
    }
  }

  public sendConfig(config: Record<string, unknown>) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'config',
        ...config,
      }));
    }
  }

  public disconnect() {
    this.isExplicitlyClosed = true;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.onStatusChangeCallback('DISCONNECTED');
  }

  private scheduleReconnect() {
    if (this.reconnectTimeout || this.isExplicitlyClosed) return;
    this.reconnectTimeout = window.setTimeout(() => {
      this.reconnectTimeout = null;
      console.log('Reconnecting WebSocket...');
      this.connect();
    }, 2500);
  }
}
