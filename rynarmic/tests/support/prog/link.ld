/* A fixed load address: the emulator drops the binary at 0x1000, so no relocation is needed. */
ENTRY(_start)
SECTIONS
{
  . = 0x1000;
  .text   : { *(.text*)   }
  .rodata : { *(.rodata*) }
  .data   : { *(.data*)   }
  .bss    : { *(.bss*) *(COMMON) }
}
