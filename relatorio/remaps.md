## Use these remaps to compile and destroy with latex

```
vim.keymap.set("n", "<leader>lc", function()
  vim.cmd("w")

  local texfile = vim.fn.expand("%")
  local pdffile = "build/" .. vim.fn.expand("%:r") .. ".pdf"

  if vim.fn.isdirectory("build") == 0 then
    vim.fn.mkdir("build", "p")
  end

  vim.fn.jobstart(
    { "latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=build", texfile },
    {
      stdout_buffered = true,
      stderr_buffered = true,
      on_exit = function(_, exit_code)
        if exit_code == 0 then
          print("✅ Compiled successfully with XeLaTeX!")
          vim.fn.jobstart({
            "cmd", "/c", "start", "", "C:\\Users\\masga\\AppData\\Local\\SumatraPDF\\SumatraPDF.exe",
            "-reuse-instance", pdffile
          }, { detach = true })
        else
          print("❌ Compilation failed. Check the log.")
        end
      end
    }
  )
end, { desc = "Compile with XeLaTeX and open PDF" })

vim.keymap.set("n", "<leader>lx", function()
  local build_dir = "build"
  local files = vim.fn.glob(build_dir .. "/*", false, true)

  for _, file in ipairs(files) do
    local ok, err = os.remove(file)
    if not ok then
      print("⚠️ Could not delete: " .. file .. " (" .. err .. ")")
    end
  end

  print("🧹 All LaTeX build files cleaned from '" .. build_dir .. "'!")
end, { desc = "Clean all LaTeX build files" })

```
