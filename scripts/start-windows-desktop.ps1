#requires -Version 7.4
[CmdletBinding()]
param(
    [ValidateRange(1024,65535)][int]$Port = 18083,
    [string]$ModelsDir,
    [string]$PreloadModel = 'Inkling-2-Mini-SQ-15-abliterated',
    [switch]$RegisterAtLogon,
    [switch]$SkipUsbCache
)

$ErrorActionPreference = 'Stop'
if ($RegisterAtLogon) {
    $pwsh = Join-Path $env:LOCALAPPDATA 'Microsoft/WindowsApps/pwsh.exe'
    if (-not (Test-Path -LiteralPath $pwsh)) { $pwsh = (Get-Process -Id $PID).Path }
    $arguments = '-NoProfile -STA -WindowStyle Hidden -File "' + $PSCommandPath + '" -Port ' + $Port
    if ($ModelsDir) { $arguments += ' -ModelsDir "' + (Resolve-Path -LiteralPath $ModelsDir).Path.TrimEnd('\') + '"' }
    if ($PreloadModel -notmatch '^[A-Za-z0-9._-]+$') { throw 'Invalid preload model ID.' }
    $arguments += ' -PreloadModel "' + $PreloadModel + '"'
    if ($SkipUsbCache) { $arguments += ' -SkipUsbCache' }
    $shell = New-Object -ComObject WScript.Shell
    foreach ($folder in 'Programs','Startup') {
        $link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath($folder)) 'Inkling2 Desktop.lnk'))
        $link.TargetPath = $pwsh
        $link.Arguments = $arguments
        $link.WorkingDirectory = $PSScriptRoot
        $link.WindowStyle = 7
        $link.Save()
    }
    return
}

Add-Type -AssemblyName PresentationFramework
$window = [Windows.Markup.XamlReader]::Parse(@'
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
 xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml" Title="Inkling2 Desktop"
 Width="790" Height="560" MinWidth="590" MinHeight="500" WindowStartupLocation="CenterScreen"
 FontFamily="Segoe UI" Background="#F4F5F4" Foreground="#192622">
 <Window.Resources>
  <Style TargetType="Button"><Setter Property="HorizontalContentAlignment" Value="Left"/>
   <Setter Property="Margin" Value="4"/><Setter Property="Padding" Value="15,11"/>
   <Setter Property="Background" Value="White"/><Setter Property="BorderBrush" Value="#C6D2CC"/>
   <Setter Property="BorderThickness" Value="1"/><Setter Property="FontSize" Value="15"/></Style>
 </Window.Resources>
 <DockPanel Margin="26,18,26,18">
  <StackPanel DockPanel.Dock="Top" Margin="4,0,4,14">
   <TextBlock Text="Inkling2 Desktop" FontSize="29" FontWeight="SemiBold"/>
   <TextBlock Text="Windows" FontSize="14" Foreground="#4D6559"/>
  </StackPanel>
  <TextBlock DockPanel.Dock="Bottom" Name="Status" TextWrapping="Wrap" Margin="4,10" FontSize="14"/>
  <StackPanel>
   <UniformGrid Columns="2">
    <Button Name="Chat" Content="Inkling2 に話す"/><Button Name="Apps" Content="インストール済みアプリ"/>
    <Button Name="Finder" Content="Finder で文書を読む"/><Button Name="Files" Content="エクスプローラー"/>
    <Button Name="Web" Content="Web 検索・閲覧"/><Button Name="Tor" Content="Tor Browser"/>
    <Button Name="Settings" Content="Windows の設定"/><Button Name="Tasks" Content="タスク マネージャー"/>
   </UniformGrid>
   <TextBlock Margin="4,19,4,4" Text="システムの状態" FontWeight="SemiBold" FontSize="17"/>
   <TextBlock Name="Telemetry" TextWrapping="Wrap" Margin="4" FontSize="14"/>
   <Button Name="Refresh" Content="状態を更新" HorizontalAlignment="Left" Padding="11,4"/>
  </StackPanel>
 </DockPanel>
</Window>
'@)
$base = "http://127.0.0.1:$Port"
$status = $window.FindName('Status')
$telemetry = $window.FindName('Telemetry')
$consoleExe = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../bin/inkling2-shell.exe'))
$tor = Join-Path $env:LOCALAPPDATA 'Programs/Tor Browser/Browser/firefox.exe'
$window.FindName('Tor').IsEnabled = Test-Path -LiteralPath $tor -PathType Leaf
$explorer = Join-Path $env:WINDIR 'explorer.exe'

function Connect-InklingDesktop {
    $listener = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    if ($listener.Count) {
        if ($listener.Count -ne 1 -or $listener[0].LocalAddress -ne '127.0.0.1' -or
            (Get-Process -Id $listener[0].OwningProcess).Path -ne $consoleExe) {
            throw '指定のポートは別のアプリが使用中です。'
        }
    } else {
        $start = @{Port=$Port; Resident=$true; PreloadModel=$PreloadModel; SkipUsbCache=$SkipUsbCache}
        if ($ModelsDir) { $start.ModelsDir = $ModelsDir }
        & (Join-Path $PSScriptRoot 'start-windows-console.ps1') @start
    }
    $health = Invoke-RestMethod "$base/health" -NoProxy -TimeoutSec 3
    if ($health.service -ne 'inkling2-os-bridge') { throw 'Inkling2 に接続できませんでした。' }
}

function Update-InklingDesktop {
    try {
        Connect-InklingDesktop
        $engine = Invoke-RestMethod "$base/v1/engine/status" -NoProxy -TimeoutSec 3
        $gpu = Invoke-RestMethod "$base/v1/gpu/info" -NoProxy -TimeoutSec 4
        $ram = Get-CimInstance Win32_OperatingSystem
        $gpuText = if ($gpu.available) { "$($gpu.name) · VRAM $($gpu.memory_used_mib) / $($gpu.memory_total_mib) MiB" } else { 'GPU 未検出' }
        $telemetry.Text = "$gpuText   /   空き RAM $([math]::Round($ram.FreePhysicalMemory / 1MB,1)) GiB"
        $status.Text = switch ($engine.state) {
            'ready' { "AI 常駐中   $($engine.model)   PID $($engine.pid)" }
            'busy'  { 'AI はモデルを読み込み中、または応答を生成中です。' }
            'idle'  { 'Inkling2 は待機中です。次の会話でモデルを読み込みます。' }
            default { 'Inkling2 は接続済みです。モデルはリクエスト時に起動します。' }
        }
    } catch { $status.Text = $_.Exception.Message }
}

$window.Add_PreviewKeyDown({ param($sender,$event)
    if ($event.Key -eq 'F5') { Update-InklingDesktop; $event.Handled=$true }
})
foreach ($entry in @{Chat='chat'; Finder='finder'; Web='browser'}.GetEnumerator()) {
    $button = $window.FindName($entry.Key)
    $button.Tag = $entry.Value
    $button.Add_Click({ param($sender,$event)
        try { Connect-InklingDesktop; Start-Process "$base/#$($sender.Tag)" }
        catch { $status.Text = $_.Exception.Message }
    })
}
$window.FindName('Apps').Add_Click({ Start-Process $explorer -ArgumentList 'shell:AppsFolder' })
$window.FindName('Files').Add_Click({ Start-Process $explorer -ArgumentList 'shell:Personal' })
$window.FindName('Settings').Add_Click({ Start-Process 'ms-settings:' })
$window.FindName('Tasks').Add_Click({ Start-Process (Join-Path ([Environment]::SystemDirectory) 'Taskmgr.exe') })
$window.FindName('Tor').Add_Click({ Start-Process -FilePath $tor -WorkingDirectory (Split-Path (Split-Path $tor)) })
$window.FindName('Refresh').Add_Click({ Update-InklingDesktop })
$timer = New-Object Windows.Threading.DispatcherTimer
$timer.Interval = [TimeSpan]::FromSeconds(15)
$timer.Add_Tick({ Update-InklingDesktop })
$window.Add_ContentRendered({ Update-InklingDesktop; $timer.Start() })
try { [void]$window.ShowDialog() } finally { $timer.Stop() }
