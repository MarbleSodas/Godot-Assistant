import { Component, signal, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { CommonModule } from '@angular/common';
import { ButtonModule } from 'primeng/button';
import { CardModule } from 'primeng/card';
import { DesktopService } from './services/desktop.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, CommonModule, ButtonModule, CardModule],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App implements OnInit {
  protected readonly title = signal('PyWebView + Angular + FastAPI');
  systemInfo = signal<any>(null);
  loading = signal<boolean>(false);

  constructor(private desktopService: DesktopService) {}

  ngOnInit() {
    // Load system info on startup if pywebview is ready
    setTimeout(() => {
      if (this.desktopService.ready) {
        this.loadSystemInfo();
      }
    }, 1000);
  }

  loadSystemInfo() {
    this.loading.set(true);
    this.desktopService.getSystemInfo().subscribe({
      next: (info) => {
        this.systemInfo.set(info);
        this.loading.set(false);
      },
      error: (error) => {
        console.error('Error loading system info:', error);
        this.loading.set(false);
      }
    });
  }
}
