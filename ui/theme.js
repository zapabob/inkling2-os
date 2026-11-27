    try {
      const saved = localStorage.getItem('inkling2.theme') || 'system';
      const dark = saved === 'dark' || (saved === 'system' && matchMedia('(prefers-color-scheme: dark)').matches);
      document.documentElement.dataset.theme = dark ? 'dark' : 'light';
    } catch {
      document.documentElement.dataset.theme = 'dark';
    }
