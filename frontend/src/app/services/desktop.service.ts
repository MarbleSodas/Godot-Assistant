import { Injectable } from '@angular/core';
import { from, Observable } from 'rxjs';

declare global {
  interface Window {
    pywebview: {
      api: {
        [key: string]: (...args: any[]) => Promise<any>;
      };
    };
  }
}

@Injectable({
  providedIn: 'root'
})
export class DesktopService {
  private isReady = false;

  constructor() {
    window.addEventListener('pywebviewready', () => {
      this.isReady = true;
      console.log('PyWebView ready!');
    });
  }

  /**
   * Call a Python method exposed via pywebview API
   * @param method The name of the Python method to call
   * @param args Arguments to pass to the Python method
   * @returns Observable that emits the result from Python
   */
  callPythonMethod(method: string, ...args: any[]): Observable<any> {
    if (!this.isReady) {
      return new Observable(observer => {
        observer.error('PyWebView not ready');
      });
    }

    return from(window.pywebview.api[method](...args));
  }

  /**
   * Get system information from Python
   */
  getSystemInfo(): Observable<any> {
    return this.callPythonMethod('get_system_info');
  }

  /**
   * Save a file using Python backend
   */
  saveFile(data: any): Observable<any> {
    return this.callPythonMethod('save_file', data);
  }

  /**
   * Check if pywebview is ready
   */
  get ready(): boolean {
    return this.isReady;
  }
}
